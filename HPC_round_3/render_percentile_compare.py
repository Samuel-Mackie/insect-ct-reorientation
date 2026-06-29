"""Render the pre-DINO / pre-rotation view for two specimens at seg thresholds p98 and p99.5.

Reuses pipeline.py's exact segment+render logic (the same mask -> zero-outside ->
render_views pass that produces the PNGs DINO sees). One canonical angle per
specimen per percentile -> 4 PNGs in HPC_round_3/percentile_compare/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import vedo
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline as P  # noqa: E402

vedo.settings.default_backend = "vtk"

# One angle for the comparison (same view for both percentiles so they line up).
ANGLE = "+X"
DIRECTION, VIEW_UP = np.array([1, 0, 0]), np.array([0, 0, 1])

SPECIMENS = [
    # ("data/original_photos/SL/soldat_10_012.tif", "soldat_10_012"),
    ("data/original_photos/BL/boffel_1_019.tif", "boffel_1_019"),
]
PERCENTILES = [98, 99.5]

# Crop each render to its non-white content (with padding) so thin/small
# specimens don't sit tiny in a large white frame.
CROP_TO_CONTENT = True
CROP_PAD = 400      # px of white margin to keep around the content

CROP_BG = 248        # pixels with all channels >= this count as background


OUT_DIR = REPO_ROOT / "HPC_round_3" / "percentile_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_to_content(path: Path) -> None:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    fg = np.any(arr < CROP_BG, axis=2)
    if not fg.any():
        return
    rows, cols = np.where(fg)
    top, bottom = rows.min(), rows.max()
    left, right = cols.min(), cols.max()
    h, w = arr.shape[:2]
    box = (max(left - CROP_PAD, 0), max(top - CROP_PAD, 0),
           min(right + CROP_PAD + 1, w), min(bottom + CROP_PAD + 1, h))
    img.crop(box).save(path)


def segment(volume: np.ndarray, pct: float) -> np.ndarray:
    t = np.percentile(volume, pct)
    mask = volume > t
    mask = ndimage.binary_dilation(mask, iterations=5)
    mask = ndimage.binary_fill_holes(mask)
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    sizes = ndimage.sum_labels(volume, labeled, index=np.arange(1, num + 1))
    return labeled == int(np.argmax(sizes) + 1)


def bbox_of_mask(mask: np.ndarray, pad: int = 10) -> tuple[slice, slice, slice]:
    """Axis-aligned voxel bounding box of a boolean mask, padded and clamped."""
    coords = np.array(np.where(mask))
    lo = np.maximum(coords.min(axis=1) - pad, 0)
    hi = np.minimum(coords.max(axis=1) + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def render_one(clean_volume: np.ndarray, out_path: Path) -> None:
    vol = vedo.Volume(clean_volume)
    xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds()
    p_min, p_max = np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])
    center = (p_min + p_max) / 2
    diag = np.linalg.norm(p_max - p_min)
    distance = max(diag * 1.5, 1.0)

    plotter = vedo.Plotter(size=P.render_size, offscreen=True, bg="white")
    try:
        plotter.show(vol, resetcam=True, zoom=P.zoom)
        cam = plotter.camera
        cam.SetFocalPoint(*center)
        cam.SetPosition(*(center + DIRECTION * distance))
        cam.SetViewUp(*VIEW_UP)
        plotter.renderer.ResetCameraClippingRange()
        plotter.render()
        plotter.screenshot(str(out_path))
    finally:
        plotter.close()


def main() -> None:
    for rel_path, name in SPECIMENS:
        in_path = REPO_ROOT / rel_path
        data = vedo.load(str(in_path)).tonumpy()

        # Segment at every percentile first, then crop the volume to the UNION
        # bounding box so both renders are framed identically (fair comparison)
        # and the specimen fills the frame at full resolution (no 2D upscaling).
        masks = {pct: segment(data, pct) for pct in PERCENTILES}
        union = np.zeros_like(next(iter(masks.values())))
        for m in masks.values():
            union |= m
        box = bbox_of_mask(union, pad=10)

        for pct, mask in masks.items():
            clean = np.zeros_like(data)
            clean[mask] = data[mask]
            clean = clean[box]
            tag = str(pct).replace(".", "_")
            out_path = OUT_DIR / f"{name}_{ANGLE}_p{tag}.png"
            render_one(clean, out_path)
            if CROP_TO_CONTENT:
                crop_to_content(out_path)
            print(f"[ok] {name}  p{pct}  kept={int(mask.sum())}  -> {out_path}")


if __name__ == "__main__":
    main()
