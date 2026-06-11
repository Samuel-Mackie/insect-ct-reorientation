from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import vedo
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from vedo import settings
from vtkmodules.vtkRenderingCore import vtkCoordinate

settings.default_backend = "vtk"


@dataclass
class RenderConfig:
    threshold_percentile: float = 97.5
    zoom: float = 1.2
    render_size: tuple[int, int] = (980, 980)
    # Match segment_original_photos so reference and query renders share scale.
    view_angle_deg: float = 25.0
    patch_size: int = 16
    patch_color: tuple[int, int, int] = (0, 180, 0)
    patch_width: int = 3

    @classmethod
    def from_metadata(
        cls,
        metadata_path: Path,
        patch_size_override: int | None = None,
    ) -> "RenderConfig":
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        cfg = meta.get("config", {})

        render_size = cfg.get("render_size", [980, 980])
        if not isinstance(render_size, list) or len(render_size) != 2:
            render_size = [980, 980]

        patch_size = patch_size_override if patch_size_override is not None else 16
        return cls(
            threshold_percentile=float(cfg.get("threshold_percentile", 97.5)),
            zoom=float(cfg.get("zoom", 1.2)),
            render_size=(int(render_size[0]), int(render_size[1])),
            patch_size=int(patch_size),
        )


VIEWS = [
    ("1", "+X", np.array([1, 0, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
    ("2", "-X", np.array([-1, 0, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
    ("3", "+Y", np.array([0, 1, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
    ("4", "-Y", np.array([0, -1, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
    ("5", "+Z", np.array([0, 0, 1], dtype=float), np.array([0, 1, 0], dtype=float)),
    ("6", "-Z", np.array([0, 0, -1], dtype=float), np.array([0, 1, 0], dtype=float)),
]


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


def patch_from_pixel(
    x: float,
    y: float,
    img_w: int,
    img_h: int,
    patch_size: int,
) -> dict[str, float | int]:
    # Clip to the cropped patch grid (img // patch_size) that top3_head_patches uses.
    # Render size (980) is divisible by 14 but not by 16, so the last partial patch
    # column/row must be excluded or prototype patches fall outside the query grid.
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


def annotation_to_world(annotation_xyz: list[int]) -> np.ndarray:
    # Annotation format is expected as [x, y, z] voxel coordinates.
    return np.array(annotation_xyz, dtype=float)


def render_with_head_marker(
    tif_path: Path,
    annotation_xyz: list[int],
    out_dir: Path,
    config: RenderConfig,
) -> dict[str, dict[str, float | str]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    volume = vedo.load(str(tif_path)).tonumpy()
    mask = segment_largest_component(volume)
    clean_data = np.zeros_like(volume)
    clean_data[mask] = volume[mask]
    vol = vedo.Volume(clean_data)

    xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds()
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2], dtype=float)
    diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    distance = max(diag * 1.5, 1.0)

    world_point = annotation_to_world(annotation_xyz)

    plotter = vedo.Plotter(size=config.render_size, offscreen=True, bg="white")
    results: dict[str, dict[str, float | str]] = {}
    try:
        for idx, label, direction, view_up in VIEWS:
            plotter.clear()
            plotter.show(vol, resetcam=True, zoom=config.zoom)

            cam = plotter.camera
            cam.SetFocalPoint(*center)
            cam.SetPosition(*(center + direction * distance))
            cam.SetViewUp(*view_up)
            cam.SetViewAngle(config.view_angle_deg)
            plotter.renderer.ResetCameraClippingRange()
            plotter.render()

            coord = vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            coord.SetValue(*world_point)
            display_x, display_y = coord.GetComputedDoubleDisplayValue(plotter.renderer)

            marked_path = out_dir / f"{tif_path.stem}_{label}_annotated_heads.png"
            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False,
                dir=out_dir,
            ) as tmp:
                tmp_path = Path(tmp.name)
            try:
                plotter.screenshot(str(tmp_path))

                img_w, img_h = config.render_size
                px = float(display_x)
                py = float(img_h - display_y)
                patch = patch_from_pixel(
                    x=px,
                    y=py,
                    img_w=img_w,
                    img_h=img_h,
                    patch_size=config.patch_size,
                )

                overlay_marker(
                    image_path=tmp_path,
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
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            results[label] = {
                "marked_image": str(marked_path),
                "head_x": round(px, 3),
                "head_y": round(py, 3),
                "image_width": img_w,
                "image_height": img_h,
                "patch_size": config.patch_size,
                "patch_col": patch["patch_col"],
                "patch_row": patch["patch_row"],
                "patch_x0": patch["patch_x0"],
                "patch_y0": patch["patch_y0"],
                "patch_x1": patch["patch_x1"],
                "patch_y1": patch["patch_y1"],
                "patch_center_x": round(float(patch["patch_center_x"]), 3),
                "patch_center_y": round(float(patch["patch_center_y"]), 3),
            }
    finally:
        plotter.close()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 6 views for annotated insect CT scans and visualize the head token patch."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("Annoteringer/annotations_output/image_annotations.json"),
        help="Path to annotation JSON.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/original_photos"),
        help="Root folder containing original TIFF volumes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/new_photos_dinov3/head_visualizations"),
        help="Root folder for rendered images with head markers.",
    )
    parser.add_argument(
        "--segmented-root",
        type=Path,
        default=Path("data/new_photos_dinov3/segmented"),
        help="Root folder that contains per-image metadata.json from segmentation.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=97.5,
        help="Fallback intensity percentile when metadata is missing.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of annotated files to process.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size used for grid visualization (16 for DINOv3 patch tokens).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)

    tasks: list[tuple[str, str, list[int], Path]] = []
    for group, group_files in annotations.items():
        for file_name, annotation in group_files.items():
            tif_path = args.input_root / group / file_name
            if tif_path.exists():
                tasks.append((group, file_name, annotation, tif_path))

    tasks.sort(key=lambda x: (x[0], x[1]))
    if args.max_files is not None:
        tasks = tasks[: max(0, args.max_files)]

    if not tasks:
        print("No matching annotated TIFF files were found.")
        return

    print(f"Found {len(tasks)} annotated files to process.")
    ok = 0
    failed = 0

    for idx, (group, file_name, annotation, tif_path) in enumerate(tasks, start=1):
        out_dir = args.output_root / group / tif_path.stem
        metadata_path = args.segmented_root / group / tif_path.stem / "metadata.json"
        if metadata_path.exists():
            cfg = RenderConfig.from_metadata(
                metadata_path=metadata_path,
                patch_size_override=args.patch_size,
            )
            config_source = str(metadata_path)
        else:
            cfg = RenderConfig(
                threshold_percentile=args.threshold_percentile,
                patch_size=args.patch_size,
            )
            config_source = "fallback_defaults"
        try:
            per_view = render_with_head_marker(
                tif_path=tif_path,
                annotation_xyz=annotation,
                out_dir=out_dir,
                config=cfg,
            )
            metadata = {
                "group": group,
                "file_name": file_name,
                "source_tif": str(tif_path),
                "annotation_xyz": annotation,
                "config": asdict(cfg),
                "config_source": config_source,
                "views": per_view,
            }
            (out_dir / "head_projection.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            ok += 1
            print(f"[{idx}/{len(tasks)}] OK   {tif_path}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{idx}/{len(tasks)}] FAIL {tif_path} ({exc})")

    print(f"Done. Successful: {ok}, Failed: {failed}, Total: {len(tasks)}")


if __name__ == "__main__":
    main()
