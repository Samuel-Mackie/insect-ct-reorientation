from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import vedo
from scipy import ndimage
from skimage.filters import threshold_multiotsu


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate original CT volumes so the fused head direction points to the "
            "target axis (+Y), AND roll about that axis so the bilateral symmetry "
            "plane becomes canonical (dorsoventral -> +X)."
        )
    )
    parser.add_argument(
        "--fused-root",
        type=Path,
        default=Path("data/new_photos/head_fused"),
        help="Root containing fused_head.json outputs.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/original_photos"),
        help="Root containing original .tif volumes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/finished_photos/rotated_symmetry"),
        help="Root for rotated outputs. Structure: <root>/<species>/{tif,json}/...",
    )
    parser.add_argument("--animal", type=str, default=None, help="Species code (e.g. AC). Omit for all.")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=97.5,
        help="Percentile for foreground segmentation used to compute center of mass.",
    )
    parser.add_argument(
        "--interp-order",
        type=int,
        default=1,
        choices=[0, 1, 3],
        help="Interpolation order for affine rotation.",
    )
    parser.add_argument(
        "--target-axis",
        type=str,
        default="+Y",
        choices=["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
        help="Axis to align head direction to. Symmetry roll only applied for +Y/-Y.",
    )
    # --- symmetry roll options ---
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="Disable the symmetry roll (behaves like the original rotate_head_up).",
    )
    parser.add_argument(
        "--symmetry-step",
        type=float,
        default=2.0,
        help="Angular resolution (degrees) for the symmetry-plane search.",
    )
    parser.add_argument(
        "--symmetry-downsample",
        type=int,
        default=2,
        help="Downsample factor for the (cheap) symmetry measurement volume.",
    )
    parser.add_argument(
        "--symmetry-conf-threshold",
        type=float,
        default=2.0,
        help="Below this confidence (contrast) the roll is flagged as unreliable.",
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Also render a finished 6-view composite (reuses segment_sixview_composite) "
        "of each rotated volume, saved under <output-root>/<species>/composite/.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rotated files.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick tests.")
    return parser.parse_args()


def discover_animals(fused_root: Path) -> list[str]:
    if not fused_root.exists():
        return []
    return sorted(p.name for p in fused_root.iterdir() if p.is_dir())


def discover_individuals(fused_root: Path, animal: str) -> list[str]:
    species_dir = fused_root / animal
    if not species_dir.exists():
        return []
    out: list[str] = []
    for p in sorted(species_dir.iterdir()):
        if (p / "json" / "fused_head.json").exists():
            out.append(p.name)
    return out


def find_original_tif(input_root: Path, animal: str, individual: str) -> Path | None:
    cands = [
        input_root / animal / f"{individual}.tif",
        input_root / animal / f"{individual}.tiff",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


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


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    b = b.astype(float)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        if c > 0.0:
            return np.eye(3, dtype=float)
        # 180-degree rotation around any axis orthogonal to a.
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = axis - np.dot(axis, a) * a
        axis /= max(np.linalg.norm(axis), 1e-12)
        x, y, z = axis
        return np.array(
            [
                [2 * x * x - 1, 2 * x * y, 2 * x * z],
                [2 * y * x, 2 * y * y - 1, 2 * y * z],
                [2 * z * x, 2 * z * y, 2 * z * z - 1],
            ],
            dtype=float,
        )

    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )
    r = np.eye(3, dtype=float) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return r


def rotation_about_y(phi_deg: float) -> np.ndarray:
    """Rotation matrix (xyz) about the +Y axis by phi_deg (the head-tail axis)."""
    p = np.deg2rad(phi_deg)
    c, s = np.cos(p), np.sin(p)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def rotate_volume(volume: np.ndarray, rot: np.ndarray, order: int) -> np.ndarray:
    # ndimage.affine_transform maps output coords -> input coords.
    # For forward rotation R around center c, use R^-1 and offset c - R^-1 c.
    center = (np.array(volume.shape, dtype=float) - 1.0) / 2.0
    r_inv = rot.T
    offset = center - r_inv @ center
    rotated = ndimage.affine_transform(
        volume,
        matrix=r_inv,
        offset=offset,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )
    return rotated


# ---------------------------------------------------------------------------
# Symmetry detection (2D cross-section: sum-projection along Y + ncc)
#   Identical method to the notebook winner: project the segmented volume along
#   the head-tail axis (Y) with "sum", then find the mirror axis that maximises
#   the normalised cross-correlation between the image and its reflection.
# ---------------------------------------------------------------------------
def reflect_image_2d(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Mirror a 2D (rows=X, cols=Z) image about the axis through the centre at
    angle_deg (same convention as the symmetry notebook)."""
    ny, nx = img.shape
    cy, cx = (np.array(img.shape) - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    X = xx - cx
    Y = yy - cy
    t = np.deg2rad(angle_deg)
    d = np.array([np.cos(t), np.sin(t)])
    n = np.array([-d[1], d[0]])
    dot = X * n[0] + Y * n[1]
    return ndimage.map_coordinates(
        img, [Y - 2 * dot * n[1] + cy, X - 2 * dot * n[0] + cx],
        order=1, mode="constant", cval=0.0,
    )


def center_image_2d(img: np.ndarray) -> np.ndarray:
    com = ndimage.center_of_mass(img > 0)
    ny, nx = img.shape
    return ndimage.shift(
        img, ((ny - 1) / 2.0 - com[0], (nx - 1) / 2.0 - com[1]),
        order=1, mode="constant", cval=0.0,
    )


def symmetry_score_2d(img: np.ndarray, angle_deg: float) -> float:
    """1 - ncc between the image and its reflection. Lower = more symmetric."""
    r = reflect_image_2d(img, angle_deg)
    num = float(np.sum(img * r))
    den = float(np.sqrt(np.sum(img ** 2) * np.sum(r ** 2))) + 1e-9
    return 1.0 - num / den


def find_symmetry_angle(arr_head: np.ndarray, step: float = 2.0) -> tuple[float, float]:
    """Find the bilateral symmetry axis of a head-up volume via the 2D cross-section.

    Projects the volume along the head-tail axis (Y = axis 1) with "sum", centres
    the image, and scans the mirror axis with the ncc metric.
    Returns (theta_deg, confidence); confidence = (median-min)/std of the score
    curve (high = clear axis, low = curved/twisted specimen)."""
    img = center_image_2d(arr_head.sum(axis=1))  # sum-projection along Y -> (X, Z)
    angles = np.arange(0.0, 180.0, step)
    scores = np.array([symmetry_score_2d(img, ang) for ang in angles])
    theta = float(angles[int(np.argmin(scores))])
    confidence = float((np.median(scores) - scores.min()) / (scores.std() + 1e-9))
    return theta, confidence


def axis_to_vector(axis: str) -> np.ndarray:
    m = {
        "+X": np.array([1.0, 0.0, 0.0], dtype=float),
        "-X": np.array([-1.0, 0.0, 0.0], dtype=float),
        "+Y": np.array([0.0, 1.0, 0.0], dtype=float),
        "-Y": np.array([0.0, -1.0, 0.0], dtype=float),
        "+Z": np.array([0.0, 0.0, 1.0], dtype=float),
        "-Z": np.array([0.0, 0.0, -1.0], dtype=float),
    }
    return m[axis]


def load_fused_head_xyz(fused_json: Path, volume_shape_xyz: tuple[int, int, int]) -> np.ndarray:
    data = json.loads(fused_json.read_text(encoding="utf-8"))
    voxel = data.get("estimated_head_xyz_voxel")
    if isinstance(voxel, list) and len(voxel) == 3:
        return np.array([float(voxel[0]), float(voxel[1]), float(voxel[2])], dtype=float)

    norm = data.get("estimated_head_xyz_norm", {})
    x = float(norm.get("x", 0.5))
    y = float(norm.get("y", 0.5))
    z = float(norm.get("z", 0.5))
    sx, sy, sz = volume_shape_xyz
    return np.array(
        [
            x * max(sx - 1, 1),
            y * max(sy - 1, 1),
            z * max(sz - 1, 1),
        ],
        dtype=float,
    )


def main() -> None:
    args = parse_args()
    animals = [args.animal] if args.animal else discover_animals(args.fused_root)
    if not animals:
        log("No animals found in fused root.")
        return

    tasks: list[tuple[str, str, Path]] = []
    for animal in animals:
        for individual in discover_individuals(args.fused_root, animal):
            fused_json = args.fused_root / animal / individual / "json" / "fused_head.json"
            tasks.append((animal, individual, fused_json))

    if args.max_files is not None:
        tasks = tasks[: max(0, args.max_files)]
    if not tasks:
        log("No fused individuals found to process.")
        return

    log(f"Found {len(tasks)} volume(s) to rotate.")
    ok = 0
    fail = 0
    flagged = 0

    # Optional: reuse the finished 6-view composite renderer from the sibling script.
    composite_fns = None
    if args.composite:
        from segment_sixview_composite import CompositeConfig, render_six_views, compose_panel
        composite_fns = (CompositeConfig(), render_six_views, compose_panel)

    target_axis = axis_to_vector(args.target_axis)
    # symmetry roll is only well-defined when the head-tail axis ends up along Y
    do_symmetry = (not args.no_symmetry) and (args.target_axis in ("+Y", "-Y"))
    if (not args.no_symmetry) and not do_symmetry:
        log(f"NOTE: symmetry roll skipped (only supported for target-axis +Y/-Y, got {args.target_axis}).")

    for idx, (animal, individual, fused_json) in enumerate(tasks, start=1):
        out_tif_dir = args.output_root / animal / "tif"
        out_json_dir = args.output_root / animal / "json"
        out_tif_dir.mkdir(parents=True, exist_ok=True)
        out_json_dir.mkdir(parents=True, exist_ok=True)
        out_tif = out_tif_dir / f"{individual}.tif"
        out_meta = out_json_dir / f"{individual}_rotation.json"

        if out_tif.exists() and not args.overwrite:
            log(f"[{idx}/{len(tasks)}] SKIP {animal}/{individual} (exists)")
            continue

        in_tif = find_original_tif(args.input_root, animal, individual)
        if in_tif is None:
            fail += 1
            log(f"[{idx}/{len(tasks)}] FAIL {animal}/{individual} (original tif not found)")
            continue

        try:
            data = vedo.load(str(in_tif)).tonumpy().astype(np.float32)
            mask = segment_largest_component(data)
            com_xyz = np.array(ndimage.center_of_mass(mask.astype(np.uint8)), dtype=float)
            head_xyz = load_fused_head_xyz(fused_json=fused_json, volume_shape_xyz=data.shape)
            v_xyz = head_xyz - com_xyz

            norm_v = float(np.linalg.norm(v_xyz))
            if norm_v < 1e-8:
                raise ValueError("Head and center-of-mass are too close; rotation axis undefined.")

            # 1) head-up rotation
            rot_head = rotation_matrix_from_vectors(v_xyz, target_axis)

            # 2) symmetry roll about Y (measured on the segmented, head-up volume)
            sym_theta: float | None = None
            sym_conf: float | None = None
            roll_deg = 0.0
            rot_total = rot_head
            if do_symmetry:
                clean = (data * mask).astype(np.float32)
                ds = max(1, args.symmetry_downsample)
                clean_small = clean[::ds, ::ds, ::ds]
                clean_head = rotate_volume(clean_small, rot=rot_head, order=1)
                sym_theta, sym_conf = find_symmetry_angle(clean_head, step=args.symmetry_step)

                # Roll about Y by a fixed formula so the symmetry plane ends up in a
                # canonical orientation (head-tail = Y, left-right = X, dorsoventral = Z).
                # roll = 90 - theta is consistent by construction across specimens.
                roll_deg = 90.0 - sym_theta
                rot_total = rotation_about_y(roll_deg) @ rot_head

            # 3) apply the combined rotation to the RAW data (single interpolation)
            rotated = rotate_volume(data, rot=rot_total, order=args.interp_order)
            rotated = rotated.astype(data.dtype, copy=False)
            tifffile.imwrite(out_tif, rotated)

            c = float(np.dot(v_xyz / norm_v, target_axis))
            c = float(np.clip(c, -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(c)))

            low_conf = bool(sym_conf is not None and sym_conf < args.symmetry_conf_threshold)
            if low_conf:
                flagged += 1

            meta = {
                "animal": animal,
                "individual": individual,
                "source_tif": str(in_tif),
                "fused_json": str(fused_json),
                "output_tif": str(out_tif),
                "threshold_percentile": args.threshold_percentile,
                "interp_order": args.interp_order,
                "target_axis": args.target_axis,
                "volume_shape_xyz": [int(data.shape[0]), int(data.shape[1]), int(data.shape[2])],
                "center_of_mass_xyz": com_xyz.tolist(),
                "head_xyz": head_xyz.tolist(),
                "head_direction_xyz": v_xyz.tolist(),
                "rotation_matrix_head": rot_head.tolist(),
                "rotation_matrix": rot_total.tolist(),  # the rotation actually applied
                "rotation_angle_deg": angle_deg,
                # --- symmetry roll info ---
                "symmetry_enabled": bool(do_symmetry),
                "symmetry_axis_deg": sym_theta,
                "symmetry_confidence": sym_conf,
                "roll_angle_deg": roll_deg,
                "low_symmetry_confidence": low_conf,
            }
            out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            # Optional finished 6-view composite of the rotated volume.
            if composite_fns is not None:
                cfg, render_six_views, compose_panel = composite_fns
                cmask = segment_largest_component(rotated)
                clean_rot = np.zeros_like(rotated)
                clean_rot[cmask] = rotated[cmask]
                title = f"{animal}/{individual}"
                if do_symmetry:
                    title += f"  roll={roll_deg:+.1f}  conf={sym_conf:.2f}"
                    if low_conf:
                        title += "  [LOW-CONF]"
                rendered = render_six_views(clean_rot.astype(float), cfg)
                panel = compose_panel(rendered, title=title, config=cfg)
                comp_dir = args.output_root / animal / "composite"
                comp_dir.mkdir(parents=True, exist_ok=True)
                panel.save(comp_dir / f"{individual}_sixview.png")

            ok += 1
            extra = ""
            if do_symmetry:
                tag = "  [LOW-CONF: check manually]" if low_conf else ""
                extra = f"  roll={roll_deg:+.1f} (sym={sym_theta:.1f}, conf={sym_conf:.2f}){tag}"
            log(f"[{idx}/{len(tasks)}] OK   {animal}/{individual} -> {out_tif}{extra}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            log(f"[{idx}/{len(tasks)}] FAIL {animal}/{individual} ({exc})")

    log(f"Done. Successful: {ok}, Failed: {fail}, Flagged (low symmetry conf): {flagged}, Total: {len(tasks)}")


if __name__ == "__main__":
    main()
