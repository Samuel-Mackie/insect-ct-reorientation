"""Method-sketch renderer: 4 images illustrating the reorientation pipeline.

For ONE individual, from ONE viewing angle, it produces:
  1_raw         the raw volume, before segmentation
  2_segmented   the same view after segmentation (largest component, bg zeroed)
  3_head_guess  the same view with a dot at the triangulated head guess
  4_head_up     the rotated volume rendered so the head points up

Everything is reused from an existing experiment run (cached segmented PNGs,
metadata, top-k head patches and the rotated `final/` volume), so DINO is NOT
re-run. Change the CONFIG block (or pass --animal/--individual/--view) to make
the same sketch for another animal.

Run from repo root:
  python HPC_round_3/method_sketch.py
  python HPC_round_3/method_sketch.py --animal AC --individual bcrick_1_000 --view +X
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import vedo
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline as P  # noqa: E402

vedo.settings.default_backend = "vtk"

# --------------------------------------------------------------------------- #
# CONFIG  (defaults = brown cricket; override on the CLI for other animals)
# --------------------------------------------------------------------------- #
ANIMAL = "AC"                 # species code (AC = brown cricket / bcrick)
INDIVIDUAL = "bcrick_1_000"   # which individual
VIEW = "+X"                   # angle for images 1-3 (one of the 6 canonical views)

# Existing run to reuse (segmented PNGs, metadata, tokens/top-k, final/ volumes).
INFO_PATH = REPO_ROOT / "data" / "test_dinov3_test_base_model"
ORIGINAL_PATH = REPO_ROOT / "data" / "original_photos"
OUT_DIR = REPO_ROOT / "HPC_round_3" / "method_sketch"

# Direction (canonical view label) to look FROM for the final head-up image.
# None -> auto-pick the first axis perpendicular to the head-up (largest) axis.
HEAD_UP_FROM = None

DOT_RADIUS = 14
DOT_COLOR = (220, 30, 30)     # red


# --------------------------------------------------------------------------- #
def render_volume_view(volume: np.ndarray, direction: np.ndarray,
                       view_up: np.ndarray, out_path: Path) -> None:
    """Render one view of a volume with the exact camera math used by the pipeline."""
    vol = vedo.Volume(volume)
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
        cam.SetPosition(*(center + np.asarray(direction, float) * distance))
        cam.SetViewUp(*np.asarray(view_up, float))
        plotter.renderer.ResetCameraClippingRange()
        plotter.render()
        plotter.screenshot(str(out_path))
    finally:
        plotter.close()


def view_vectors(angle: str) -> tuple[np.ndarray, np.ndarray]:
    """Look up (direction, view_up) for a canonical view label from pipeline.VIEWS."""
    for label, direction, vup in P.VIEWS:
        if label == angle:
            return np.asarray(direction, float), np.asarray(vup, float)
    raise ValueError(f"Unknown view '{angle}'. Options: {[v[0] for v in P.VIEWS]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=ANIMAL)
    ap.add_argument("--individual", default=INDIVIDUAL)
    ap.add_argument("--view", default=VIEW)
    args = ap.parse_args()

    animal, individual, view = args.animal, args.individual, args.view
    seg_root = INFO_PATH / "segmented"
    tokens_root = INFO_PATH / "tokens"
    seg_dir = seg_root / animal / individual

    out_dir = OUT_DIR / individual
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    meta = json.loads((seg_dir / "metadata.json").read_text(encoding="utf-8"))

    # --- Image 1: raw volume, before segmentation -------------------------- #
    raw = vedo.load(str(ORIGINAL_PATH / animal / f"{individual}.tif")).tonumpy()
    direction, view_up = view_vectors(view)
    render_volume_view(raw, direction, view_up, out_dir / f"1_raw_{view}.png")
    print(f"[1] raw            -> 1_raw_{view}.png")

    # --- Image 2: after segmentation (reuse cached render) ----------------- #
    cached_seg = seg_dir / f"{individual}_{view}.png"
    img2 = out_dir / f"2_segmented_{view}.png"
    shutil.copyfile(cached_seg, img2)
    print(f"[2] segmented      -> 2_segmented_{view}.png  (cached render)")

    # --- Image 3: head guess (triangulate, then dot on the same view) ------ #
    origins, dirs = P.build_rays_from_individual(seg_root, tokens_root, animal, individual)
    fused = P.ransac_fuse(origins, dirs, threshold=P.ransac_threshold,
                          refine_iters=P.ransac_iterations)
    head_xyz = np.asarray(fused["point"], float)

    cam = {v["angle"]: v for v in meta["render_geometry"]}[view]
    px, py = P.project_point_to_pixel(
        head_xyz,
        np.array(cam["camera_position"], float),
        np.array(cam["camera_focal_point"], float),
        np.array(cam["view_up"], float),
        P.image_w, P.image_h, float(cam["fov_y_deg"]),
    )
    img = Image.open(cached_seg).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse([px - DOT_RADIUS, py - DOT_RADIUS, px + DOT_RADIUS, py + DOT_RADIUS],
                 fill=DOT_COLOR, outline=(0, 0, 0), width=2)
    img.save(out_dir / f"3_head_guess_{view}.png")
    print(f"[3] head guess     -> 3_head_guess_{view}.png  "
          f"(pixel=({px:.0f},{py:.0f}), inliers={fused['n_inliers']})")

    # --- Image 4: rotated volume, head pointing up ------------------------- #
    # Rotate in the consistent [x,y,z] frame ourselves (the saved final/ tif is
    # stored transposed, which scrambles the axis mapping on reload). Align the
    # head->COM direction onto the largest axis, render perpendicular with that
    # axis up, so the head ends at the top.
    target_axis = ["+X", "+Y", "+Z"][int(np.argmax(meta["volume_shape"]))]
    up_vec = P.AXIS_TO_VECTOR[target_axis]
    from_axis = HEAD_UP_FROM or next(a for a in ("+X", "+Y", "+Z") if a != target_axis)
    look_dir = P.AXIS_TO_VECTOR[from_axis]

    com_xyz = np.array(meta["center_of_mass_xyz"], float)
    mask = P.segment_largest_component(raw)
    clean = np.zeros_like(raw)
    clean[mask] = raw[mask]
    rot = P.rotation_matrix_from_vectors(head_xyz - com_xyz, up_vec)
    rotated = P.rotate_volume(clean, rot=rot, order=1, center=com_xyz)

    render_volume_view(rotated, look_dir, up_vec, out_dir / f"4_head_up_{from_axis}.png")
    print(f"[4] head up        -> 4_head_up_{from_axis}.png  "
          f"(head axis={target_axis}, viewed from {from_axis})")

    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
