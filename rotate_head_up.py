from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import vedo
from scipy import ndimage


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate original CT volumes so fused head direction points upward (+Z)."
    )
    parser.add_argument(
        "--fused-root",
        type=Path,
        default=Path("data/new_photos/head_fused_v1"),
        help="Root containing fused_head_v1.json outputs.",
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
        default=Path("data/finished_photos/rotated"),
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
        help="Axis to align head direction to. +Y usually corresponds to visual 'up' in +Z view.",
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
        if (p / "json" / "fused_head_v1.json").exists():
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


def segment_largest_component(volume: np.ndarray, threshold_percentile: float) -> np.ndarray:
    mask = volume > np.percentile(volume, threshold_percentile)
    structure = np.ones((3, 3, 3), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure)
    mask = ndimage.binary_closing(mask, structure=structure)
    mask = ndimage.binary_fill_holes(mask)

    labeled, num = ndimage.label(mask)
    if num == 0:
        raise ValueError("No connected components found after thresholding.")
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    largest = int(np.argmax(sizes) + 1)
    return labeled == largest


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
            fused_json = args.fused_root / animal / individual / "json" / "fused_head_v1.json"
            tasks.append((animal, individual, fused_json))

    if args.max_files is not None:
        tasks = tasks[: max(0, args.max_files)]
    if not tasks:
        log("No fused individuals found to process.")
        return

    log(f"Found {len(tasks)} volume(s) to rotate.")
    ok = 0
    fail = 0

    target_axis = axis_to_vector(args.target_axis)
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
            mask = segment_largest_component(data, threshold_percentile=args.threshold_percentile)
            com_xyz = np.array(ndimage.center_of_mass(mask.astype(np.uint8)), dtype=float)
            head_xyz = load_fused_head_xyz(fused_json=fused_json, volume_shape_xyz=data.shape)
            v_xyz = head_xyz - com_xyz

            norm_v = float(np.linalg.norm(v_xyz))
            if norm_v < 1e-8:
                raise ValueError("Head and center-of-mass are too close; rotation axis undefined.")

            rot = rotation_matrix_from_vectors(v_xyz, target_axis)
            rotated = rotate_volume(data, rot=rot, order=args.interp_order)
            rotated = rotated.astype(data.dtype, copy=False)
            tifffile.imwrite(out_tif, rotated)

            c = float(np.dot(v_xyz / norm_v, target_axis))
            c = float(np.clip(c, -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(c)))
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
                "rotation_matrix": rot.tolist(),
                "rotation_angle_deg": angle_deg,
            }
            out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            ok += 1
            log(f"[{idx}/{len(tasks)}] OK   {animal}/{individual} -> {out_tif}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            log(f"[{idx}/{len(tasks)}] FAIL {animal}/{individual} ({exc})")

    log(f"Done. Successful: {ok}, Failed: {fail}, Total: {len(tasks)}")


if __name__ == "__main__":
    main()
