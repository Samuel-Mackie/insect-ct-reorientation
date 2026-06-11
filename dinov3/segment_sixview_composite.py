from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import vedo
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from vedo import settings

settings.default_backend = "vtk"


@dataclass
class CompositeConfig:
    threshold_percentile: float = 97.5
    zoom: float = 1.2
    tile_size: tuple[int, int] = (980, 980)
    background: str = "white"
    canvas_bg: tuple[int, int, int] = (255, 255, 255)
    panel_padding: int = 24
    label_height: int = 36
    label_color: tuple[int, int, int] = (20, 20, 20)
    label_font_size: int = 34


VIEWS = [
    ("1", "+X", np.array([1, 0, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
    ("2", "-X", np.array([-1, 0, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
    ("3", "+Y", np.array([0, 1, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
    ("4", "-Y", np.array([0, -1, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
    ("5", "+Z", np.array([0, 0, 1], dtype=float), np.array([0, 1, 0], dtype=float)),
    ("6", "-Z", np.array([0, 0, -1], dtype=float), np.array([0, 1, 0], dtype=float)),
]

# Fixed standard locations in the final 2x3 panel.
# "Head-up" (+Z) is always top-left.
PANEL_ORDER = [
    "+Z", "+Y", "+X",
    "-X", "-Y", "-Z",
]


def find_input_volumes(input_root: Path) -> list[Path]:
    # Preferred layout: <root>/<species>/tif/*.tif
    preferred = sorted(input_root.glob("*/*/*.tif"))
    preferred = [p for p in preferred if p.parent.name.lower() == "tif"]
    if preferred:
        return preferred
    return sorted([p for p in input_root.rglob("*.tif") if p.is_file()])


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


def render_six_views(clean_volume: np.ndarray, config: CompositeConfig) -> dict[str, Image.Image]:
    vol = vedo.Volume(clean_volume)
    xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds()
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2], dtype=float)
    diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    distance = max(diag * 1.5, 1.0)

    images: dict[str, Image.Image] = {}
    plotter = vedo.Plotter(size=config.tile_size, offscreen=True, bg=config.background)
    try:
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            for idx, label, direction, view_up in VIEWS:
                plotter.clear()
                plotter.show(vol, resetcam=True, zoom=config.zoom)

                cam = plotter.camera
                cam.SetFocalPoint(*center)
                cam.SetPosition(*(center + direction * distance))
                cam.SetViewUp(*view_up)
                plotter.renderer.ResetCameraClippingRange()

                shot = tdir / f"tmp_{label}.png"
                plotter.screenshot(str(shot))
                images[label] = Image.open(shot).convert("RGB")
    finally:
        plotter.close()
    return images


def compose_panel(views: dict[str, Image.Image], title: str, config: CompositeConfig) -> Image.Image:
    tile_w, tile_h = config.tile_size
    rows, cols = 2, 3
    pad = config.panel_padding
    label_h = config.label_height

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * (tile_h + label_h) + (rows + 1) * pad + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=config.canvas_bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", config.label_font_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text((pad, pad // 2), title, fill=config.label_color, font=font)

    for i, angle in enumerate(PANEL_ORDER):
        r, c = divmod(i, cols)
        x = pad + c * (tile_w + pad)
        y = pad + label_h + r * (tile_h + label_h + pad)
        img = views[angle]
        canvas.paste(img, (x, y))
        # Label on panel header.
        draw.text((x, y - 28), angle, fill=config.label_color, font=font)
        # Label overlay on the image itself.
        box_w, box_h = 140, 52
        bx0, by0 = x + 10, y + 10
        bx1, by1 = bx0 + box_w, by0 + box_h
        draw.rectangle((bx0, by0, bx1, by1), fill=(255, 255, 255))
        draw.rectangle((bx0, by0, bx1, by1), outline=(40, 40, 40), width=2)
        draw.text((bx0 + 16, by0 + 10), angle, fill=(10, 10, 10), font=font)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment CT volumes and save one fixed-layout 6-view composite image per volume."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/finished_photos_dinov3/rotated"),
        help="Input root, typically data/finished_photos_dinov3/rotated with <species>/tif/*.tif.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/finished_photos_dinov3/composite"),
        help="Output root for one-image composites.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=97.5,
        help="Intensity percentile used for segmentation threshold.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing composites.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CompositeConfig(threshold_percentile=args.threshold_percentile)
    volumes = find_input_volumes(args.input_root)
    if args.max_files is not None:
        volumes = volumes[: max(0, args.max_files)]

    if not volumes:
        print(f"No .tif files found under {args.input_root}", flush=True)
        return

    print(f"Found {len(volumes)} volume(s)", flush=True)
    ok = 0
    fail = 0
    for i, in_path in enumerate(volumes, start=1):
        species = in_path.parent.parent.name if in_path.parent.name.lower() == "tif" else in_path.parent.name
        out_species = args.output_root / species
        out_img_dir = out_species / "images"
        out_json_dir = out_species / "json"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_json_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_img_dir / f"{in_path.stem}_sixview.png"
        out_json = out_json_dir / f"{in_path.stem}_sixview.json"

        if out_png.exists() and not args.overwrite:
            print(f"[{i}/{len(volumes)}] SKIP {in_path}", flush=True)
            continue

        try:
            data = vedo.load(str(in_path)).tonumpy()
            mask = segment_largest_component(data)
            clean = np.zeros_like(data)
            clean[mask] = data[mask]

            rendered = render_six_views(clean, cfg)
            panel = compose_panel(rendered, title=f"{species}/{in_path.stem}", config=cfg)
            panel.save(out_png)

            meta = {
                "source_tif": str(in_path),
                "output_image": str(out_png),
                "species": species,
                "panel_order": PANEL_ORDER,
                "config": asdict(cfg),
                "volume_shape_zyx": [int(data.shape[0]), int(data.shape[1]), int(data.shape[2])],
                "kept_voxels": int(mask.sum()),
                "total_voxels": int(mask.size),
            }
            out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            ok += 1
            print(f"[{i}/{len(volumes)}] OK   {in_path} -> {out_png}", flush=True)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[{i}/{len(volumes)}] FAIL {in_path} ({exc})", flush=True)

    print(f"Done. Successful: {ok}, Failed: {fail}, Total: {len(volumes)}", flush=True)


if __name__ == "__main__":
    main()
