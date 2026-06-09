from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import vedo
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from vedo import settings

settings.default_backend = "vtk"


@dataclass
class SegmentationConfig:
    threshold_percentile: float = 97.5
    zoom: float = 1.2
    render_size: tuple[int, int] = (980, 980)
    # Vertical camera view angle (FOV) explicitly enforced per view so the
    # perspective is deterministic and recorded for downstream reprojection.
    # VTK default is 30 deg; zoom=1.2 historically shrank it to 30/1.2 = 25 deg.
    view_angle_deg: float = 25.0


VIEWS = [
    ("1", "+X", np.array([1, 0, 0]), np.array([0, 0, 1])),
    ("2", "-X", np.array([-1, 0, 0]), np.array([0, 0, 1])),
    ("3", "+Y", np.array([0, 1, 0]), np.array([0, 0, 1])),
    ("4", "-Y", np.array([0, -1, 0]), np.array([0, 0, 1])),
    ("5", "+Z", np.array([0, 0, 1]), np.array([0, 1, 0])),
    ("6", "-Z", np.array([0, 0, -1]), np.array([0, 1, 0])),
]


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


def render_views(
    clean_volume: np.ndarray,
    out_dir: Path,
    base_name: str,
    config: SegmentationConfig,
) -> dict[str, object]:
    vol = vedo.Volume(clean_volume)
    xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds()
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2], dtype=float)
    diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    distance = max(diag * 1.5, 1.0)
    camera_views: list[dict[str, object]] = []

    plotter = vedo.Plotter(size=config.render_size, offscreen=True, bg="white")
    try:
        for idx, label, direction, view_up in VIEWS:
            plotter.clear()
            plotter.show(vol, resetcam=True, zoom=config.zoom)

            cam = plotter.camera
            cam.SetFocalPoint(*center)
            cam.SetPosition(*(center + direction * distance))
            cam.SetViewUp(*view_up)
            # Enforce a fixed view angle so the perspective does not depend on
            # vedo's zoom (which divides the view angle and can compound across
            # views as the camera is reused).
            cam.SetViewAngle(config.view_angle_deg)
            plotter.renderer.ResetCameraClippingRange()

            out_path = out_dir / f"{base_name}_{idx}_{label}.png"
            plotter.screenshot(str(out_path))
            cam_pos = (center + direction * distance).tolist()
            camera_views.append(
                {
                    "view_index": int(idx),
                    "angle": label,
                    "direction": direction.astype(float).tolist(),
                    "view_up": view_up.astype(float).tolist(),
                    "camera_position": cam_pos,
                    "camera_focal_point": center.astype(float).tolist(),
                    "view_angle_deg": float(cam.GetViewAngle()),
                    "output_image": str(out_path),
                }
            )
    finally:
        plotter.close()

    return {
        "bounds_xyzxyz": [float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)],
        "volume_center_xyz": center.astype(float).tolist(),
        "camera_distance": float(distance),
        "view_angle_deg": float(config.view_angle_deg),
        "camera_views": camera_views,
    }


def process_volume(in_path: Path, in_root: Path, out_root: Path, config: SegmentationConfig) -> Path:
    rel_parent = in_path.parent.relative_to(in_root)
    out_dir = out_root / rel_parent / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    data = vedo.load(str(in_path)).tonumpy()
    mask = segment_largest_component(data)

    clean_data = np.zeros_like(data)
    clean_data[mask] = data[mask]
    render_meta = render_views(clean_data, out_dir=out_dir, base_name=in_path.stem, config=config)

    # Voxel axis convention for numpy volume arrays from vedo in this project: [x, y, z].
    shape_xyz = [int(data.shape[0]), int(data.shape[1]), int(data.shape[2])]
    shape_zyx = [shape_xyz[2], shape_xyz[1], shape_xyz[0]]

    metadata = {
        "source_path": str(in_path),
        "output_dir": str(out_dir),
        "config": asdict(config),
        "volume_shape": shape_xyz,  # Backward-compatible alias; treat as xyz.
        "volume_shape_zyx": shape_zyx,
        "volume_shape_xyz": shape_xyz,
        "voxel_axis_order": "xyz",
        "kept_voxels": int(mask.sum()),
        "total_voxels": int(mask.size),
        "mask_fill_ratio": float(mask.sum() / max(mask.size, 1)),
        "render_geometry": render_meta,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def find_input_volumes(in_root: Path) -> list[Path]:
    exts = {".tif", ".tiff"}
    return sorted(p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment CT volumes and render 6 canonical screenshots per input file."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/original_photos"),
        help="Root folder containing original .tif/.tiff files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/new_photos/segmented"),
        help="Root folder for segmented screenshots.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=97.5,
        help="Intensity percentile used for binary thresholding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even when target folder already has rendered PNGs.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of input volumes to process (useful for quick tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_root = args.input_root
    out_root = args.output_root

    if not in_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {in_root}")

    config = SegmentationConfig(threshold_percentile=args.threshold_percentile)
    volumes = find_input_volumes(in_root)
    if not volumes:
        print(f"No TIFF files found under {in_root}")
        return
    if args.max_files is not None:
        volumes = volumes[: max(0, args.max_files)]
        if not volumes:
            print("No files selected after applying --max-files.")
            return

    print(f"Found {len(volumes)} volumes under {in_root}")
    ok = 0
    failed = 0

    for idx, in_path in enumerate(volumes, start=1):
        rel_parent = in_path.parent.relative_to(in_root)
        out_dir = out_root / rel_parent / in_path.stem
        existing_pngs = sorted(out_dir.glob("*.png")) if out_dir.exists() else []
        if existing_pngs and not args.overwrite:
            print(f"[{idx}/{len(volumes)}] SKIP {in_path} (already rendered)")
            continue

        try:
            rendered_dir = process_volume(in_path, in_root, out_root, config)
            ok += 1
            print(f"[{idx}/{len(volumes)}] OK   {in_path} -> {rendered_dir}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{idx}/{len(volumes)}] FAIL {in_path} ({exc})")

    print(f"Done. Successful: {ok}, Failed: {failed}, Total: {len(volumes)}")


if __name__ == "__main__":
    main()
