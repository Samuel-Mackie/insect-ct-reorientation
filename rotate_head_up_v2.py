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
        description="Rotate CT volumes with two-axis alignment (head axis + body-side axis)."
    )
    parser.add_argument(
        "--fused-root",
        type=Path,
        default=Path("data/new_photos/head_fused_v2"),
        help="Root containing fused_head_v2.json outputs.",
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
        default=Path("data/finished_photos/v2/rotated"),
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
        "--target-up-axis",
        type=str,
        default="+Y",
        choices=["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
        help="Primary target axis for head direction.",
    )
    parser.add_argument(
        "--target-side-axis",
        type=str,
        default="+X",
        choices=["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
        help="Secondary target axis for roll stabilization.",
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
        if (p / "json" / "fused_head_v2.json").exists() or (p / "json" / "fused_head_v1.json").exists():
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


def rotate_volume(volume: np.ndarray, rot: np.ndarray, order: int) -> np.ndarray:
    center = (np.array(volume.shape, dtype=float) - 1.0) / 2.0
    r_inv = rot.T
    offset = center - r_inv @ center
    return ndimage.affine_transform(
        volume,
        matrix=r_inv,
        offset=offset,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )


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


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Vector norm too small for normalization.")
    return v / n


def project_orthogonal(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return v - np.dot(v, axis) * axis


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


def find_fused_json(fused_root: Path, animal: str, individual: str) -> Path | None:
    p2 = fused_root / animal / individual / "json" / "fused_head_v2.json"
    if p2.exists():
        return p2
    p1 = fused_root / animal / individual / "json" / "fused_head_v1.json"
    if p1.exists():
        return p1
    return None


def pca_axes(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)  # xyz
    if coords.shape[0] < 10:
        raise ValueError("Too few foreground voxels for PCA-based secondary axis.")

    centered = coords.astype(float) - coords.astype(float).mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vecs[:, order]  # columns by descending variance


def choose_secondary_axis_stable(
    pcs: np.ndarray,
    primary_src: np.ndarray,
    up_tgt: np.ndarray,
    side_tgt: np.ndarray,
    prev_forward_rot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # target forward axis from right-handed target frame
    forward_tgt = normalize_vec(np.cross(side_tgt, up_tgt))

    long_src = project_orthogonal(pcs[:, 0], primary_src)
    if np.linalg.norm(long_src) < 1e-8:
        long_src = project_orthogonal(pcs[:, 2], primary_src)
    long_src = normalize_vec(long_src)

    cand_base = [pcs[:, 1], pcs[:, 2]]
    best_s = None
    best_forward = None
    best_score = -1e18

    for base in cand_base:
        s0 = project_orthogonal(base, primary_src)
        if np.linalg.norm(s0) < 1e-8:
            continue
        s0 = normalize_vec(s0)
        for sign in (1.0, -1.0):
            s = s0 * sign
            rot = two_axis_rotation(
                primary_src=primary_src,
                secondary_src=s,
                primary_tgt=up_tgt,
                secondary_tgt=side_tgt,
            )
            forward_rot = normalize_vec(rot @ long_src)
            score = float(np.dot(forward_rot, forward_tgt))
            if prev_forward_rot is not None:
                score += 0.75 * float(np.dot(forward_rot, prev_forward_rot))
            if score > best_score:
                best_score = score
                best_s = s
                best_forward = forward_rot

    if best_s is None or best_forward is None:
        raise ValueError("Could not derive stable secondary axis from PCA candidates.")
    return best_s, best_forward


def make_frame(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    y = normalize_vec(primary)
    x = normalize_vec(project_orthogonal(secondary, y))
    z = normalize_vec(np.cross(x, y))
    x = normalize_vec(np.cross(y, z))
    return np.column_stack([x, y, z])


def two_axis_rotation(
    primary_src: np.ndarray,
    secondary_src: np.ndarray,
    primary_tgt: np.ndarray,
    secondary_tgt: np.ndarray,
) -> np.ndarray:
    src = make_frame(primary_src, secondary_src)
    tgt = make_frame(primary_tgt, secondary_tgt)
    return tgt @ src.T


def main() -> None:
    args = parse_args()
    animals = [args.animal] if args.animal else discover_animals(args.fused_root)
    if not animals:
        log("No animals found in fused root.")
        return

    up_tgt = normalize_vec(axis_to_vector(args.target_up_axis))
    side_tgt = axis_to_vector(args.target_side_axis)
    side_tgt = normalize_vec(project_orthogonal(side_tgt, up_tgt))
    if float(np.linalg.norm(side_tgt)) < 1e-8:
        raise ValueError("target-side-axis is parallel to target-up-axis; choose a different side axis.")

    tasks: list[tuple[str, str, Path]] = []
    for animal in animals:
        for individual in discover_individuals(args.fused_root, animal):
            fused_json = find_fused_json(args.fused_root, animal, individual)
            if fused_json is not None:
                tasks.append((animal, individual, fused_json))

    if args.max_files is not None:
        tasks = tasks[: max(0, args.max_files)]
    if not tasks:
        log("No fused individuals found to process.")
        return

    log(f"Found {len(tasks)} volume(s) to rotate.")
    ok = 0
    fail = 0

    prev_forward_by_animal: dict[str, np.ndarray] = {}

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
            primary_src = head_xyz - com_xyz
            primary_src = normalize_vec(primary_src)

            pcs = pca_axes(mask)
            prev_forward = prev_forward_by_animal.get(animal)
            secondary_src, forward_rot = choose_secondary_axis_stable(
                pcs=pcs,
                primary_src=primary_src,
                up_tgt=up_tgt,
                side_tgt=side_tgt,
                prev_forward_rot=prev_forward,
            )
            prev_forward_by_animal[animal] = forward_rot

            rot = two_axis_rotation(
                primary_src=primary_src,
                secondary_src=secondary_src,
                primary_tgt=up_tgt,
                secondary_tgt=side_tgt,
            )

            rotated = rotate_volume(data, rot=rot, order=args.interp_order).astype(data.dtype, copy=False)
            tifffile.imwrite(out_tif, rotated)

            c = float(np.clip(np.dot(primary_src, up_tgt), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(c)))

            meta = {
                "animal": animal,
                "individual": individual,
                "source_tif": str(in_tif),
                "fused_json": str(fused_json),
                "output_tif": str(out_tif),
                "mode": "two_axis",
                "threshold_percentile": args.threshold_percentile,
                "interp_order": args.interp_order,
                "target_up_axis": args.target_up_axis,
                "target_side_axis": args.target_side_axis,
                "volume_shape_xyz": [int(data.shape[0]), int(data.shape[1]), int(data.shape[2])],
                "center_of_mass_xyz": com_xyz.tolist(),
                "head_xyz": head_xyz.tolist(),
                "primary_source_axis_xyz": primary_src.tolist(),
                "secondary_source_axis_xyz": secondary_src.tolist(),
                "forward_axis_after_rotation_xyz": forward_rot.tolist(),
                "target_up_vector_xyz": up_tgt.tolist(),
                "target_side_vector_xyz": side_tgt.tolist(),
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
