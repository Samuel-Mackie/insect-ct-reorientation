"""
Importable copy of the full_pipeline_dinov3.ipynb functions.

Why this file exists: process-based parallelism on Windows uses 'spawn', which
re-imports the worker function by reference. Functions defined in notebook cells
cannot be pickled that way, so the heavy, independent steps live here instead.
The notebook is unchanged; run_parallel.py imports from this module.

torch / transformers are imported lazily inside the DINO functions only, so the
render and rotation worker processes never pay the (large) import cost.
"""
from pathlib import Path
import math
import json

import numpy as np
import vedo
from vedo import settings
from PIL import Image
from skimage.filters import threshold_otsu
from scipy import ndimage
from sklearn.metrics.pairwise import cosine_similarity
import tifffile

settings.default_backend = "vtk"

# ----------------------------------------------------------------------------
# Config (mirrors the notebook; patch_size/grid fixed for dinov3-vitb16)
# ----------------------------------------------------------------------------
# label, direction, view_up
VIEWS = [
    ("+X", np.array([1, 0, 0]), np.array([0, 0, 1])),
    ("-X", np.array([-1, 0, 0]), np.array([0, 0, 1])),
    ("+Y", np.array([0, 1, 0]), np.array([0, 0, 1])),
    ("-Y", np.array([0, -1, 0]), np.array([0, 0, 1])),
    ("+Z", np.array([0, 0, 1]), np.array([0, 1, 0])),
    ("-Z", np.array([0, 0, -1]), np.array([0, 1, 0])),
]

AXIS_TO_VECTOR = {
    "+X": np.array([1.0, 0.0, 0.0], dtype=float),
    "-X": np.array([-1.0, 0.0, 0.0], dtype=float),
    "+Y": np.array([0.0, 1.0, 0.0], dtype=float),
    "-Y": np.array([0.0, -1.0, 0.0], dtype=float),
    "+Z": np.array([0.0, 0.0, 1.0], dtype=float),
    "-Z": np.array([0.0, 0.0, -1.0], dtype=float),
}

animals = ["AC", "BC", "BF", "BL", "BP", "CF", "GH", "MA", "PP", "WO"]

model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"

zoom = 1.2
patch_size = 16
grid_w = 60
grid_h = 60
image_w = grid_w * patch_size
image_h = grid_h * patch_size
render_size = (image_w, image_h)

ransac_threshold = 10
ransac_iterations = 4

# Almost white
bg_threshold = 240
# Ratio of foreground to background to include patch
fg_patch_ratio = 0.2

# CLS + register tokens to skip; set for real when the DINO model is loaded.
# dinov3-vitb16 has 4 register tokens, so 1 + 4 = 5.
NUM_PREFIX_TOKENS = 5


# ----------------------------------------------------------------------------
# Segmentation + rendering
# ----------------------------------------------------------------------------
def find_input_volumes(in_root: Path) -> list[Path]:
    # Finds all .tif files in folders and subfolders of input
    return sorted(in_root.rglob("*.tif"))


def segment_largest_component(volume: np.ndarray) -> np.ndarray:
    threshold = threshold_otsu(volume)
    mask = volume > threshold

    mask = ndimage.binary_dilation(mask, iterations=5)
    mask = ndimage.binary_fill_holes(mask)

    labeled, num = ndimage.label(mask)

    # Weighted sum of the components
    sizes = ndimage.sum_labels(volume, labeled, index=np.arange(1, num + 1))

    # Finds largest weight and return
    largest = int(np.argmax(sizes) + 1)
    mask = labeled == largest
    return mask


def render_views(clean_volume: np.ndarray, out_dir: Path, base_name: str) -> list:
    vol = vedo.Volume(clean_volume)
    # Locate corners
    xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds()
    p_min, p_max = np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])

    # Calculate camera position
    center = (p_min + p_max) / 2
    diag = np.linalg.norm(p_max - p_min)
    distance = max(diag * 1.5, 1.0)
    camera_views: list[dict[str, object]] = []

    plotter = vedo.Plotter(size=render_size, offscreen=True, bg="white")
    try:
        plotter.show(vol, resetcam=True, zoom=zoom)
        cam = plotter.camera

        for label, direction, view_up in VIEWS:
            cam.SetFocalPoint(*center)
            cam.SetPosition(*(center + direction * distance))
            cam.SetViewUp(*view_up)
            plotter.renderer.ResetCameraClippingRange()
            plotter.render()
            out_path = out_dir / f"{base_name}_{label}.png"
            plotter.screenshot(str(out_path))

            fov_y_deg = float(cam.GetViewAngle())
            cam_pos = (center + direction * distance).tolist()
            camera_views.append(
                {
                    "angle":              label,
                    "view_up":            view_up.astype(float).tolist(),
                    "camera_position":    cam_pos,
                    "camera_focal_point": center.astype(float).tolist(),
                    "fov_y_deg":          fov_y_deg,
                }
            )
    finally:
        plotter.close()

    return camera_views


def process_volume(in_path: Path, in_root: Path, out_root: Path) -> Path:
    # Finds the species from the file path and builds the output path
    rel_parent = in_path.parent.relative_to(in_root)
    out_dir = out_root / rel_parent / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    data = vedo.load(str(in_path)).tonumpy()
    mask = segment_largest_component(data)

    # Center of mass of the segmented animal (used later as the rotation center).
    com_xyz = [float(c) for c in ndimage.center_of_mass(mask)]

    clean_data = np.zeros_like(data)
    clean_data[mask] = data[mask]
    render_meta = render_views(clean_data, out_dir=out_dir, base_name=in_path.stem)

    # Voxel axis convention for numpy volume arrays from vedo in this project: [x, y, z].
    shape_xyz = [int(data.shape[0]), int(data.shape[1]), int(data.shape[2])]

    metadata = {
        "source_path": str(in_path),
        "output_dir": str(out_dir),
        "volume_shape": shape_xyz,
        "kept_voxels": int(mask.sum()),
        "total_voxels": int(mask.size),
        "center_of_mass_xyz": com_xyz,
        "render_geometry": render_meta,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


# ----------------------------------------------------------------------------
# Head projection
# ----------------------------------------------------------------------------
def camera_basis(cam_pos: np.ndarray, focal: np.ndarray, view_up: np.ndarray):
    # Gets right, up and front unit vectors for a camera position
    f = focal - cam_pos
    f = f / np.linalg.norm(f)
    u0 = view_up / np.linalg.norm(view_up)
    r = np.cross(f, u0)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    u = u / np.linalg.norm(u)
    return r, u, f


def project_point_to_pixel(point_xyz, cam_pos, focal, view_up, image_w, image_h, fov_y_deg):
    """
    Forward perspective projection: world point -> pixel (top-left origin, y down).
    Exact inverse of build_ray_from_pixel.
    """
    r, u, f = camera_basis(cam_pos, focal, view_up)

    w = point_xyz - cam_pos

    depth = float(np.dot(w, f))
    x_cam = float(np.dot(w, r))
    y_cam = float(np.dot(w, u))

    tan_half = math.tan(math.radians(fov_y_deg) * 0.5)

    x_ndc = (x_cam / depth) / tan_half
    y_ndc = (y_cam / depth) / tan_half

    nx = x_ndc * 0.5 + 0.5
    ny = 0.5 - y_ndc * 0.5

    return nx * image_w, ny * image_h


def get_head_information(in_path: Path, in_root: Path, out_root: Path, annotation_xyz: list) -> dict:
    rel_parent = in_path.parent.relative_to(in_root)
    out_dir = out_root / rel_parent / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    camera_views = metadata["render_geometry"]

    world_point = annotation_xyz
    out: dict[str, dict[str, object]] = {}

    for view in camera_views:
        label = view["angle"]
        cam_pos = np.array(view["camera_position"], dtype=float)
        focal = np.array(view["camera_focal_point"], dtype=float)
        view_up = np.array(view["view_up"], dtype=float)
        fov_y = float(view["fov_y_deg"])

        px, py = project_point_to_pixel(world_point, cam_pos, focal, view_up, image_w, image_h, fov_y)

        out[label] = {
            "patch_col": int(px // patch_size),
            "patch_row": int(py // patch_size),
        }

    (out_dir / "head_projection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------
# DINOv3 patch tokens  (torch imported lazily; only the main process uses these)
# ----------------------------------------------------------------------------
def load_dino_model():
    """Load processor + model once in the main process. Sets NUM_PREFIX_TOKENS."""
    global NUM_PREFIX_TOKENS
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINO model={model_name} | device={device}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    NUM_PREFIX_TOKENS = 1 + num_register_tokens  # CLS + register tokens
    print(f"Model loaded | num_prefix_tokens={NUM_PREFIX_TOKENS}")
    return processor, model, device


def extract_patch_tokens(image_path: Path, segmented_root: Path, tokens_root: Path,
                         processor, model, device: str) -> None:
    import torch

    rel = image_path.relative_to(segmented_root)
    token_dir = tokens_root / rel.parent / image_path.stem
    token_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = token_dir / "tokens.npy"
    if tokens_path.exists():
        return None

    image = Image.open(image_path).convert("RGB")

    with torch.inference_mode():
        inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
        pixel_values = inputs["pixel_values"].to(device)
        outputs = model(pixel_values)
        patch_tokens = outputs.last_hidden_state[:, NUM_PREFIX_TOKENS:, :].squeeze(0).cpu().numpy()

    np.save(tokens_path, patch_tokens)
    return None


def discover_segmented_individuals(segmented_root: Path, animal: str) -> list[str]:
    species_dir = segmented_root / animal
    if not species_dir.exists():
        return []
    return sorted(p.name for p in species_dir.iterdir() if p.is_dir())


def build_prototype_animal(animal: str, tokens_root: Path, segmented_root: Path):
    proto_list = []

    for individual in discover_segmented_individuals(segmented_root, animal):
        head_info_path = segmented_root / animal / individual / "head_projection.json"
        if not head_info_path.exists():
            continue

        head_info = json.loads(head_info_path.read_text(encoding="utf-8"))

        for angle, angle_info in head_info.items():
            patch_row = int(angle_info["patch_row"])
            patch_col = int(angle_info["patch_col"])

            tokens_path = tokens_root / animal / individual / f"{individual}_{angle}" / "tokens.npy"
            if not tokens_path.exists():
                continue

            tokens = np.load(tokens_path, mmap_mode="r")  # (n_patches, embed_dim)
            idx = patch_row * grid_w + patch_col
            proto_list.append(tokens[idx].copy())

    if not proto_list:
        raise ValueError(f"No prototype vectors could be built for animal '{animal}'")

    prototype_vector = np.mean(proto_list, axis=0)
    np.save(tokens_root / animal / "prototype_vector.npy", prototype_vector)
    return prototype_vector


# ----------------------------------------------------------------------------
# Cosine similarity -> top-k head patches
# ----------------------------------------------------------------------------
def make_foreground_patch_mask(image: Image.Image, bg_threshold: int, fg_patch_ratio: float) -> np.ndarray:
    img = np.array(image)

    # Foreground pixels: all RGB values under the threshold
    fg_pixel = np.all(img < bg_threshold, axis=2)

    # Reshape into patches and compute foreground fraction per patch
    patches = fg_pixel.reshape(grid_h, patch_size, grid_w, patch_size)
    fg_fraction = patches.mean(axis=(1, 3))
    fg_patch = fg_fraction > fg_patch_ratio
    return fg_patch.ravel()


def save_top_k_patches(tokens_root: Path, segmented_root: Path, animal: str, k: int):
    prototype_vector = np.load(tokens_root / animal / "prototype_vector.npy").reshape(1, -1)

    for individual in discover_segmented_individuals(segmented_root, animal):
        individual_dir = tokens_root / animal / individual

        for angle_dir in sorted(individual_dir.iterdir()):
            if not angle_dir.is_dir():
                continue

            tokens_path = angle_dir / "tokens.npy"
            img_path = segmented_root / animal / individual / f"{angle_dir.name}.png"
            out_path = angle_dir / "top_k_patches.json"

            if not tokens_path.exists() or not img_path.exists():
                continue

            tokens = np.load(tokens_path)
            image = Image.open(img_path).convert("RGB")
            fg_flat = make_foreground_patch_mask(image, bg_threshold, fg_patch_ratio)

            fg_indices = np.where(fg_flat)[0]
            if len(fg_indices) == 0:
                continue

            fg_sims = cosine_similarity(tokens[fg_indices], prototype_vector).flatten()

            actual_k = min(k, len(fg_indices))
            top_order = np.argsort(fg_sims)[::-1][:actual_k]
            top_flat = fg_indices[top_order]

            patches = [
                {"patch_row": int(flat_idx // grid_w), "patch_col": int(flat_idx % grid_w)}
                for flat_idx in top_flat
            ]

            out_path.write_text(json.dumps({"patches": patches}, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# Rays + triangulation
# ----------------------------------------------------------------------------
def build_ray_from_pixel(px, py, cam_pos, focal, view_up, fov_y_deg):
    # Normalize to [0,1]
    nx = px / float(image_w)
    ny = py / float(image_h)
    # Normalized device coordinates in [-1,1] (image y increases down, so flip)
    x_ndc = (nx - 0.5) * 2.0
    y_ndc = -((ny - 0.5) * 2.0)
    tan_half = np.tan(np.deg2rad(fov_y_deg) * 0.5)

    d_cam = np.array([x_ndc * tan_half, y_ndc * tan_half, 1.0], dtype=float)
    d_cam = d_cam / np.linalg.norm(d_cam)

    r, u, f = camera_basis(cam_pos, focal, view_up)
    d_world = d_cam[0] * r + d_cam[1] * u + d_cam[2] * f
    d_world = d_world / np.linalg.norm(d_world)
    return cam_pos, d_world


def load_camera_by_angle(segmented_meta: Path) -> dict:
    d = json.loads(segmented_meta.read_text(encoding="utf-8"))
    return {v.get("angle"): v for v in d.get("render_geometry", [])}


def triangulate_rays(origins: list, dirs: list) -> np.ndarray:
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    for o, u in zip(origins, dirs):
        M = np.eye(3) - np.outer(u, u)
        A += M
        b += M @ o
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return p


def point_ray_distance(p: np.ndarray, o: np.ndarray, u: np.ndarray) -> float:
    d = p - o
    return float(np.linalg.norm(d - np.dot(u, d) * u))


def build_rays_from_individual(segmented_root: Path, tokens_root: Path, animal: str, individual: str):
    cam_by_angle = load_camera_by_angle(segmented_root / animal / individual / "metadata.json")

    origins, dirs = [], []
    for angle, *_ in VIEWS:
        if angle not in cam_by_angle:
            continue

        cam = cam_by_angle[angle]
        cam_pos = np.array(cam["camera_position"], dtype=float)
        focal = np.array(cam["camera_focal_point"], dtype=float)
        view_up = np.array(cam["view_up"], dtype=float)
        fov_y = float(cam["fov_y_deg"])

        json_path = tokens_root / animal / individual / f"{individual}_{angle}" / "top_k_patches.json"
        if not json_path.exists():
            continue

        for patch in json.loads(json_path.read_text(encoding="utf-8"))["patches"]:
            # Patch center in pixels from row/col (true center)
            px = (patch["patch_col"] + 0.5) * patch_size
            py = (patch["patch_row"] + 0.5) * patch_size
            origin, direction = build_ray_from_pixel(px, py, cam_pos, focal, view_up, fov_y)
            origins.append(origin)
            dirs.append(direction)

    return origins, dirs


def ransac_fuse(origins: list, dirs: list, threshold: float, refine_iters: int):
    def inliers_of(p: np.ndarray) -> list:
        return [k for k in range(len(origins)) if point_ray_distance(p, origins[k], dirs[k]) <= threshold]

    best = None
    # Initial fit over all ray pairs
    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            if np.array_equal(origins[i], origins[j]):
                continue
            p = triangulate_rays([origins[i], origins[j]], [dirs[i], dirs[j]])
            inliers = inliers_of(p)
            if best is None or len(inliers) > best[0]:
                best = (len(inliers), p, inliers)

    if best is None:
        return None

    # Refinement
    _, p, inliers = best
    for _ in range(max(0, refine_iters)):
        if len(inliers) < 2:
            break
        p = triangulate_rays([origins[k] for k in inliers], [dirs[k] for k in inliers])
        inliers = inliers_of(p)

    return {"point": p, "inliers": inliers, "n_inliers": len(inliers), "threshold": threshold}


# ----------------------------------------------------------------------------
# Rotation
# ----------------------------------------------------------------------------
def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    b = b.astype(float)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= np.linalg.norm(b)

    v = np.cross(a, b)
    c = float(np.dot(a, b))        # cos between a and b
    s = float(np.linalg.norm(v))   # sin between a and b

    # Check if already parallel
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)

        # 180-degree rotation around any axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, a) * a
        axis /= max(np.linalg.norm(axis), 1e-12)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def rotate_volume(volume: np.ndarray, rot: np.ndarray, order: int, center: np.ndarray) -> np.ndarray:
    r_inv = rot.T
    offset = center - r_inv @ center
    return ndimage.affine_transform(
        volume, matrix=r_inv, offset=offset, order=order,
        mode="constant", cval=0.0, prefilter=order > 1,
    )
