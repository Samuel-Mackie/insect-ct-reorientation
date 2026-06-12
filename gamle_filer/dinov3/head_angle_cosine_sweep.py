from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import vedo
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from transformers import AutoImageProcessor, AutoModel
from vedo import settings
from vtkmodules.vtkRenderingCore import vtkCoordinate

settings.default_backend = "vtk"


@dataclass
class SweepRenderConfig:
    threshold_percentile: float = 97.5
    zoom: float = 1.2
    render_size: tuple[int, int] = (980, 980)
    patch_size: int = 16
    patch_color: tuple[int, int, int] = (0, 180, 0)
    patch_width: int = 3


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one annotated CT volume around an orbit and compute cosine similarity "
            "between each projected head patch and one reference-angle head patch."
        )
    )
    parser.add_argument(
        "--animal",
        type=str,
        default=None,
        help="Species code (e.g. AC). Used with --file-name or to label direct --tif-path runs.",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="TIFF file name/stem under input-root/<animal>. If omitted, use --tif-path.",
    )
    parser.add_argument(
        "--tif-path",
        type=Path,
        default=None,
        help="Direct path to one TIFF volume. Use with --annotation-xyz or a matching annotation JSON entry.",
    )
    parser.add_argument(
        "--annotation-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Annotated head voxel position in xyz order. Overrides --annotations lookup.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("Annoteringer/annotations_output/image_annotations.json"),
        help="Annotation JSON used when --annotation-xyz is not provided.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/original_photos"),
        help="Root folder containing original .tif/.tiff volumes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/new_photos_dinov3/head_angle_cosine_sweep"),
        help="Root output folder.",
    )
    parser.add_argument(
        "--reference-angle",
        type=float,
        default=0.0,
        help="Reference orbit angle in degrees. Must match one generated angle.",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=10.0,
        help="Orbit angle step in degrees.",
    )
    parser.add_argument(
        "--start-angle",
        type=float,
        default=0.0,
        help="First orbit angle in degrees.",
    )
    parser.add_argument(
        "--end-angle",
        type=float,
        default=360.0,
        help="Stop angle in degrees, exclusive.",
    )
    parser.add_argument(
        "--orbit-axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="Axis to orbit around. Default z gives an xy-plane sweep.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face DINO model id.",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/cuda:0.")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=97.5,
        help="Intensity percentile used for foreground segmentation.",
    )
    parser.add_argument("--zoom", type=float, default=1.2, help="Vedo camera zoom.")
    parser.add_argument(
        "--render-size",
        type=int,
        nargs=2,
        default=(980, 980),
        metavar=("WIDTH", "HEIGHT"),
        help="Rendered image size.",
    )
    parser.add_argument("--bg-threshold", type=int, default=240, help="White-background threshold.")
    parser.add_argument(
        "--fg-patch-ratio",
        type=float,
        default=0.20,
        help="Minimum foreground pixel ratio in a patch.",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> dict[str, dict[str, list[int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, list[int]]] = {}
    for group, files in data.items():
        group_dict: dict[str, list[int]] = {}
        for name, coord in files.items():
            if isinstance(coord, list) and len(coord) == 3:
                group_dict[name] = [int(coord[0]), int(coord[1]), int(coord[2])]
        out[group] = group_dict
    return out


def find_input_tif(input_root: Path, animal: str, file_name: str) -> Path:
    stem_or_name = Path(file_name)
    candidates: list[Path] = []
    if stem_or_name.suffix:
        candidates.append(input_root / animal / stem_or_name.name)
    else:
        candidates.append(input_root / animal / f"{stem_or_name.name}.tif")
        candidates.append(input_root / animal / f"{stem_or_name.name}.tiff")

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No TIFF found for {animal}/{file_name} under {input_root}")


def find_annotation(
    annotations: dict[str, dict[str, list[int]]],
    animal: str | None,
    file_name: str,
) -> tuple[str, list[int]]:
    name = Path(file_name).name
    if animal is not None:
        animal_entries = annotations.get(animal, {})
        if name in animal_entries:
            return animal, animal_entries[name]
        stem_name = f"{Path(name).stem}.tif"
        if stem_name in animal_entries:
            return animal, animal_entries[stem_name]
        raise KeyError(f"No annotation found for {animal}/{name}")

    matches: list[tuple[str, list[int]]] = []
    for group, files in annotations.items():
        if name in files:
            matches.append((group, files[name]))
        else:
            stem_name = f"{Path(name).stem}.tif"
            if stem_name in files:
                matches.append((group, files[stem_name]))

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"No annotation found for {name}")
    groups = [m[0] for m in matches]
    raise ValueError(f"Annotation name is ambiguous across groups: {groups}")


def resolve_inputs(args: argparse.Namespace) -> tuple[str, str, Path, list[float]]:
    if args.tif_path is None:
        if args.animal is None or args.file_name is None:
            raise ValueError("Use either --tif-path or both --animal and --file-name.")
        tif_path = find_input_tif(args.input_root, args.animal, args.file_name)
        animal = args.animal
        file_name = tif_path.name
    else:
        tif_path = args.tif_path
        if not tif_path.exists():
            raise FileNotFoundError(f"TIFF does not exist: {tif_path}")
        animal = args.animal or tif_path.parent.name
        file_name = tif_path.name

    if args.annotation_xyz is not None:
        annotation = [float(v) for v in args.annotation_xyz]
    else:
        annotations = load_annotations(args.annotations)
        animal, annotation_int = find_annotation(annotations, animal, file_name)
        annotation = [float(v) for v in annotation_int]

    return animal, Path(file_name).stem, tif_path, annotation


def segment_largest_component(volume: np.ndarray) -> np.ndarray:
    thresholds = threshold_multiotsu(volume, classes=3)
    regions = np.digitize(volume, bins=thresholds)
    mask = regions == 2
    mask = ndimage.binary_dilation(mask, iterations=5)
    mask = ndimage.binary_fill_holes(mask)
    labeled, num = ndimage.label(mask)
    sizes = ndimage.sum_labels(volume, labeled, index=np.arange(1, num + 1))
    largest = int(np.argmax(sizes) + 1)
    mask = labeled == largest
    return mask


def angle_values(start: float, end: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("--angle-step must be > 0.")
    if end <= start:
        raise ValueError("--end-angle must be greater than --start-angle.")

    values: list[float] = []
    angle = float(start)
    while angle < end - 1e-9:
        values.append(round(angle, 6))
        angle += step
    if not values:
        raise ValueError("No angles generated.")
    return values


def circular_angle_diff(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def find_reference_angle(angles: list[float], reference_angle: float) -> float:
    ref = reference_angle % 360.0
    diffs = [circular_angle_diff(a % 360.0, ref) for a in angles]
    idx = int(np.argmin(diffs))
    if diffs[idx] > 1e-6:
        raise ValueError(
            f"--reference-angle {reference_angle} is not in the generated angle list. "
            f"Try a multiple of --angle-step."
        )
    return angles[idx]


def angle_label(angle: float) -> str:
    normalized = angle % 360.0
    if abs(normalized - round(normalized)) < 1e-6:
        return f"{int(round(normalized)):03d}"
    label = f"{normalized:.3f}".rstrip("0").rstrip(".")
    return label.replace(".", "p")


def orbit_direction(angle_deg: float, axis: str) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    if axis == "x":
        direction = np.array([0.0, np.cos(theta), np.sin(theta)], dtype=float)
    elif axis == "y":
        direction = np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=float)
    else:
        direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    return direction / max(np.linalg.norm(direction), 1e-12)


def orbit_view_up(axis: str) -> np.ndarray:
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([0.0, 0.0, 1.0], dtype=float)


def patch_from_pixel(
    x: float,
    y: float,
    img_w: int,
    img_h: int,
    patch_size: int,
) -> dict[str, float | int]:
    # Clip to the cropped patch grid (img // patch_size); render size 980 is not
    # divisible by 16, so the last partial patch column/row must be excluded.
    col = int(np.clip(x // patch_size, 0, img_w // patch_size - 1))
    row = int(np.clip(y // patch_size, 0, img_h // patch_size - 1))

    x0 = col * patch_size
    y0 = row * patch_size
    x1 = min(x0 + patch_size - 1, img_w - 1)
    y1 = min(y0 + patch_size - 1, img_h - 1)

    center_x = x0 + (x1 - x0 + 1) / 2.0
    center_y = y0 + (y1 - y0 + 1) / 2.0

    return {
        "patch_col": col,
        "patch_row": row,
        "patch_x0": int(x0),
        "patch_y0": int(y0),
        "patch_x1": int(x1),
        "patch_y1": int(y1),
        "patch_center_x": float(center_x),
        "patch_center_y": float(center_y),
    }


def overlay_marker(
    image_path: Path,
    out_path: Path,
    patch_bounds: tuple[int, int, int, int],
    patch_color: tuple[int, int, int],
    patch_width: int,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(patch_bounds, outline=patch_color, width=patch_width)
    img.save(out_path)


def render_orbit_views(
    clean_volume: np.ndarray,
    annotation_xyz: list[float],
    angles: list[float],
    orbit_axis: str,
    out_dir: Path,
    base_name: str,
    config: SweepRenderConfig,
) -> dict[float, dict[str, object]]:
    image_dir = out_dir / "images"
    viz_dir = out_dir / "visualisations"
    image_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    vol = vedo.Volume(clean_volume)
    xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds()
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2], dtype=float)
    diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    distance = max(diag * 1.5, 1.0)
    world_point = np.array(annotation_xyz, dtype=float)
    view_up = orbit_view_up(orbit_axis)
    img_w, img_h = config.render_size

    results: dict[float, dict[str, object]] = {}
    plotter = vedo.Plotter(size=config.render_size, offscreen=True, bg="white")
    try:
        for idx, angle in enumerate(angles, start=1):
            label = angle_label(angle)
            direction = orbit_direction(angle, orbit_axis)
            camera_position = center + direction * distance

            plotter.clear()
            plotter.show(vol, resetcam=True, zoom=config.zoom)

            cam = plotter.camera
            cam.SetFocalPoint(*center)
            cam.SetPosition(*camera_position)
            cam.SetViewUp(*view_up)
            plotter.renderer.ResetCameraClippingRange()
            plotter.render()

            coord = vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            coord.SetValue(*world_point)
            display_x, display_y = coord.GetComputedDoubleDisplayValue(plotter.renderer)
            px = float(display_x)
            py = float(img_h - display_y)

            patch = patch_from_pixel(
                x=px,
                y=py,
                img_w=img_w,
                img_h=img_h,
                patch_size=config.patch_size,
            )

            image_path = image_dir / f"{base_name}_angle_{label}.png"
            marked_path = viz_dir / f"{base_name}_angle_{label}_head_patch.png"
            plotter.screenshot(str(image_path))
            overlay_marker(
                image_path=image_path,
                out_path=marked_path,
                patch_bounds=(
                    int(patch["patch_x0"]),
                    int(patch["patch_y0"]),
                    int(patch["patch_x1"]),
                    int(patch["patch_y1"]),
                ),
                patch_color=config.patch_color,
                patch_width=config.patch_width,
            )

            results[angle] = {
                "angle_index": idx,
                "angle_deg": float(angle),
                "direction": direction.tolist(),
                "view_up": view_up.tolist(),
                "camera_position": camera_position.tolist(),
                "camera_focal_point": center.tolist(),
                "image": str(image_path),
                "marked_image": str(marked_path),
                "head_x": round(px, 3),
                "head_y": round(py, 3),
                "image_width": img_w,
                "image_height": img_h,
                **patch,
            }
    finally:
        plotter.close()

    return results


def crop_to_patch_multiple(image: Image.Image, patch_size: int) -> Image.Image:
    w, h = image.size
    nw = (w // patch_size) * patch_size
    nh = (h // patch_size) * patch_size
    if nw <= 0 or nh <= 0:
        raise ValueError(f"Image too small for patch_size={patch_size}: {(w, h)}")
    if (nw, nh) == (w, h):
        return image
    return image.crop((0, 0, nw, nh))


def make_foreground_patch_mask(
    image: Image.Image,
    patch_size: int,
    bg_threshold: int,
    fg_patch_ratio: float,
) -> tuple[np.ndarray, int, int]:
    img = np.asarray(image)
    h, w = img.shape[:2]
    gh, gw = h // patch_size, w // patch_size
    bg = np.all(img > bg_threshold, axis=2)
    fg_pixel = ~bg
    fg_patch = np.zeros((gh, gw), dtype=bool)
    for r in range(gh):
        for c in range(gw):
            p = fg_pixel[r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size]
            fg_patch[r, c] = p.mean() > fg_patch_ratio
    return fg_patch.reshape(-1), gh, gw


def extract_patch_tokens(
    image_path: Path,
    processor: AutoImageProcessor,
    model: AutoModel,
    patch_size: int,
    bg_threshold: int,
    fg_patch_ratio: float,
    device: str,
) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    image = crop_to_patch_multiple(image, patch_size)
    fg_flat, gh, gw = make_foreground_patch_mask(image, patch_size, bg_threshold, fg_patch_ratio)

    with torch.inference_mode():
        inputs = processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # DINOv3: skip CLS + register tokens (1 + num_register_tokens) to keep patch tokens.
        num_prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))
        tokens = outputs.last_hidden_state[:, num_prefix:, :].squeeze(0).detach().cpu().numpy()

    if tokens.shape[0] != gh * gw:
        raise ValueError(f"Patch count mismatch in {image_path}: got {tokens.shape[0]}, expected {gh*gw}")
    return {"image": image, "tokens": tokens, "fg_flat": fg_flat, "grid_h": gh, "grid_w": gw}


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-8)


def patch_token_from_projection(
    entry: dict[str, object],
    features: dict[str, object],
    patch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    tokens = features["tokens"]  # type: ignore[assignment]
    fg_flat = features["fg_flat"]  # type: ignore[assignment]
    grid_h = int(features["grid_h"])  # type: ignore[arg-type]
    grid_w = int(features["grid_w"])  # type: ignore[arg-type]
    image: Image.Image = features["image"]  # type: ignore[assignment]

    patch = patch_from_pixel(
        x=float(entry["head_x"]),
        y=float(entry["head_y"]),
        img_w=image.width,
        img_h=image.height,
        patch_size=patch_size,
    )
    row = int(patch["patch_row"])
    col = int(patch["patch_col"])
    if not (0 <= row < grid_h and 0 <= col < grid_w):
        raise ValueError(f"Projected patch outside token grid: row={row}, col={col}, grid={grid_h}x{grid_w}")

    idx = row * grid_w + col
    patch_info = {
        **patch,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "foreground_patch": bool(fg_flat[idx]),  # type: ignore[index]
    }
    return tokens[idx], patch_info  # type: ignore[index]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "angle_deg",
        "similarity",
        "patch_row",
        "patch_col",
        "patch_center_x",
        "patch_center_y",
        "foreground_patch",
        "image",
        "marked_image",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def save_similarity_plot(path: Path, rows: list[dict[str, object]], reference_angle: float) -> None:
    angles = [float(row["angle_deg"]) for row in rows]
    sims = [float(row["similarity"]) for row in rows]

    plt.figure(figsize=(10, 4))
    plt.plot(angles, sims, marker="o", linewidth=1.5)
    plt.axvline(reference_angle, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Cosine similarity")
    plt.ylim(-1.0, 1.0)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    animal, individual, tif_path, annotation = resolve_inputs(args)

    angles = angle_values(args.start_angle, args.end_angle, args.angle_step)
    reference_angle = find_reference_angle(angles, args.reference_angle)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Starting angle cosine sweep | model={args.model_name} | device={device}")
    log("Loading DINO processor...")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    log("Loading DINO model...")
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    log("Model loaded.")

    patch_size = int(getattr(model.config, "patch_size", 16))
    config = SweepRenderConfig(
        threshold_percentile=args.threshold_percentile,
        zoom=args.zoom,
        render_size=(int(args.render_size[0]), int(args.render_size[1])),
        patch_size=patch_size,
    )

    base_dir = args.output_root / animal / individual
    data_dir = base_dir / "data"
    json_dir = base_dir / "json"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading volume: {tif_path}")
    data = vedo.load(str(tif_path)).tonumpy()
    mask = segment_largest_component(data)
    clean = np.zeros_like(data)
    clean[mask] = data[mask]

    log(f"Rendering {len(angles)} angle(s) around {args.orbit_axis}-axis")
    render_entries = render_orbit_views(
        clean_volume=clean,
        annotation_xyz=annotation,
        angles=angles,
        orbit_axis=args.orbit_axis,
        out_dir=base_dir,
        base_name=individual,
        config=config,
    )

    log("Extracting projected head patch tokens...")
    patch_tokens: dict[float, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for idx, angle in enumerate(angles, start=1):
        entry = render_entries[angle]
        log(f"  Token {idx}/{len(angles)}: angle {angle:g}")
        features = extract_patch_tokens(
            image_path=Path(str(entry["image"])),
            processor=processor,
            model=model,
            patch_size=patch_size,
            bg_threshold=args.bg_threshold,
            fg_patch_ratio=args.fg_patch_ratio,
            device=device,
        )
        token, patch_info = patch_token_from_projection(entry, features, patch_size)
        patch_tokens[angle] = token
        entry.update(patch_info)

    reference_token = patch_tokens[reference_angle].reshape(1, -1)
    reference_token = l2_normalize(reference_token, axis=1)

    for angle in angles:
        token = l2_normalize(patch_tokens[angle].reshape(1, -1), axis=1)
        similarity = float((token @ reference_token.T).squeeze())
        entry = render_entries[angle]
        row = {
            "angle_deg": float(angle),
            "similarity": similarity,
            "is_reference": bool(angle == reference_angle),
            "image": entry["image"],
            "marked_image": entry["marked_image"],
            "head_x": entry["head_x"],
            "head_y": entry["head_y"],
            "patch_row": entry["patch_row"],
            "patch_col": entry["patch_col"],
            "patch_x0": entry["patch_x0"],
            "patch_y0": entry["patch_y0"],
            "patch_x1": entry["patch_x1"],
            "patch_y1": entry["patch_y1"],
            "patch_center_x": entry["patch_center_x"],
            "patch_center_y": entry["patch_center_y"],
            "foreground_patch": entry["foreground_patch"],
            "direction": entry["direction"],
            "view_up": entry["view_up"],
            "camera_position": entry["camera_position"],
            "camera_focal_point": entry["camera_focal_point"],
        }
        rows.append(row)

    stem = f"{individual}_ref_{angle_label(reference_angle)}_angle_cosine_sweep"
    csv_path = data_dir / f"{stem}.csv"
    npy_path = data_dir / f"{stem}.npy"
    json_path = json_dir / f"{stem}.json"
    plot_path = base_dir / "visualisations" / f"{stem}.png"

    write_csv(csv_path, rows)
    np.save(npy_path, np.array([[float(r["angle_deg"]), float(r["similarity"])] for r in rows], dtype=float))
    save_similarity_plot(plot_path, rows, reference_angle=reference_angle)

    payload = {
        "animal": animal,
        "individual": individual,
        "source_tif": str(tif_path),
        "annotation_xyz": annotation,
        "model_name": args.model_name,
        "device": device,
        "orbit_axis": args.orbit_axis,
        "start_angle": float(args.start_angle),
        "end_angle": float(args.end_angle),
        "angle_step": float(args.angle_step),
        "reference_angle": float(reference_angle),
        "config": asdict(config),
        "volume_shape_xyz": [int(data.shape[0]), int(data.shape[1]), int(data.shape[2])],
        "kept_voxels": int(mask.sum()),
        "total_voxels": int(mask.size),
        "csv": str(csv_path),
        "similarity_array": str(npy_path),
        "plot": str(plot_path),
        "angles": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best = max(rows, key=lambda r: float(r["similarity"]))
    log(f"Saved CSV:  {csv_path}")
    log(f"Saved JSON: {json_path}")
    log(f"Saved plot: {plot_path}")
    log(f"Best angle: {float(best['angle_deg']):g} (cosine={float(best['similarity']):.4f})")


if __name__ == "__main__":
    main()
