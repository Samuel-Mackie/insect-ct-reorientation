"""Render one specimen from the 4 pyramid views and the 6 canonical views.

These are the rendered views fed to DINO (segmented volume, before rotation).
Top block: 4 tetrahedral "pyramid" views (P1-P4). Bottom block: 6 canonical
axis views (+/-X, +/-Y, +/-Z). Output is one combined PNG.

Run from repo root:
  python HPC_round_3/view_overview.py
  python HPC_round_3/view_overview.py --animal AC --individual bcrick_1_000
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import vedo
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline as P  # noqa: E402

vedo.settings.default_backend = "vtk"

# --------------------------------------------------------------------------- #
ANIMAL = "AC"
INDIVIDUAL = "bcrick_1_001"
ORIGINAL_PATH = REPO_ROOT / "data" / "original_photos"
OUT_DIR = REPO_ROOT / "HPC_round_3" / "view_overview"

# 6 canonical views (label, direction, view_up) -- straight from the pipeline.
VIEWS_6 = [
    ("+X", np.array([1, 0, 0]), np.array([0, 0, 1])),
    ("-X", np.array([-1, 0, 0]), np.array([0, 0, 1])),
    ("+Y", np.array([0, 1, 0]), np.array([0, 0, 1])),
    ("-Y", np.array([0, -1, 0]), np.array([0, 0, 1])),
    ("+Z", np.array([0, 0, 1]), np.array([0, 1, 0])),
    ("-Z", np.array([0, 0, -1]), np.array([0, 1, 0])),
]


def _pyramid_up(direction: np.ndarray) -> np.ndarray:
    d = direction / np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(d, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    return up


# 4 tetrahedral "pyramid" views.
_PYRAMID_DIRS = [
    ("P1", np.array([1.0, 1.0, 1.0])),
    ("P2", np.array([1.0, -1.0, -1.0])),
    ("P3", np.array([-1.0, 1.0, -1.0])),
    ("P4", np.array([-1.0, -1.0, 1.0])),
]
VIEWS_4 = [(lbl, d / np.linalg.norm(d), _pyramid_up(d)) for lbl, d in _PYRAMID_DIRS]


def render_volume_view(volume, direction, view_up, out_path: Path) -> None:
    vol = vedo.Volume(volume)
    xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds()
    p_min, p_max = np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])
    center = (p_min + p_max) / 2
    distance = max(np.linalg.norm(p_max - p_min) * 1.5, 1.0)

    plotter = vedo.Plotter(size=P.render_size, offscreen=True, bg="white")
    try:
        plotter.show(vol, resetcam=True, zoom=P.zoom)
        cam = plotter.camera
        cam.SetFocalPoint(*center)
        cam.SetPosition(*(center + np.asarray(direction, float) * distance))
        cam.SetViewUp(*np.asarray(view_up, float))
        plotter.renderer.ResetCameraClippingRange()
        plotter.render()
        plotter.screenshot(str(out_path))
    finally:
        plotter.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=ANIMAL)
    ap.add_argument("--individual", default=INDIVIDUAL)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = vedo.load(str(ORIGINAL_PATH / args.animal / f"{args.individual}.tif")).tonumpy()
    mask = P.segment_largest_component(raw)
    clean = np.zeros_like(raw)
    clean[mask] = raw[mask]

    tmp = Path(tempfile.mkdtemp())
    rendered = {}
    for label, direction, vup in VIEWS_4 + VIEWS_6:
        out = tmp / f"{label}.png"
        render_volume_view(clean, direction, vup, out)
        rendered[label] = mpimg.imread(out)
        print(f"[ok] rendered {label}")

    # Two separate files: one for the 4 pyramid views, one for the 6 canonical.
    fig4, ax4 = plt.subplots(2, 2, figsize=(6, 6))
    fig4.suptitle("4 pyramid views (tetrahedral)", fontsize=13, fontweight="bold")
    for ax, (label, *_), in zip(ax4.ravel(), VIEWS_4):
        ax.imshow(rendered[label]); ax.set_title(label, fontsize=11); ax.axis("off")
    out4 = OUT_DIR / f"{args.individual}_4views.png"
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close(fig4)

    fig6, ax6 = plt.subplots(2, 3, figsize=(9, 6))
    fig6.suptitle("6 canonical views (axis-aligned)", fontsize=13, fontweight="bold")
    for ax, (label, *_), in zip(ax6.ravel(), VIEWS_6):
        ax.imshow(rendered[label]); ax.set_title(label, fontsize=11); ax.axis("off")
    out6 = OUT_DIR / f"{args.individual}_6views.png"
    fig6.savefig(out6, dpi=150, bbox_inches="tight")
    plt.close(fig6)

    print(f"\nDone -> {out4}\n        {out6}")


if __name__ == "__main__":
    main()
