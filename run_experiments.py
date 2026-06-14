"""
Experiment harness for the insect CT reorientation pipeline.

Goal: measure TIMING (per pipeline pass) and CORRECTNESS for a set of parameter
variations, so we can argue the best time/error trade-off for each knob.

Knobs tested (each described in Danish in the project brief):
  - antal vinkler            6 vs 4 camera views
  - model size               small / base / large / huge  (DINOv3 ViT-S/B/L/H+)
  - patches / resolution     grid 30 / 60 / 90  (render = grid * 16 px)
  - segmentation threshold   otsu vs percentile 96 / 97 / 98  (background cutoff)
  - top-k head patches        3 / 5 / 8
  - number of annotations    1 image / 1 animal / 5 animals  (prototype source)

Individuals: the dataset has many individuals per species, named
<prefix>_<group>_<NNN>.tif with many scan groups (gras_1_*, gras_17_*, ...).
The EVALUATION set -- the ones rotated and turned into composites -- are the
individuals that (a) share a prefix with an annotated individual, so they belong
to the prototype's id family (e.g. gras_1_*), AND (b) whose trailing index _NNN
is in --eval-range (default 11-20). The prefix match is essential: without it
every group's _011.._020 would match and hundreds of unrelated individuals would
be processed. The annotated individuals (e.g. gras_1_000..004) are rendered and
tokenized to build the prototype, even when their index is outside the range,
but they are not rotated/composited themselves.

Design: one-factor-at-a-time (OFAT). A baseline config is held fixed while one
axis is swept at a time. That is 14 configs instead of 2592 (full grid) and maps
directly onto "discuss the best choice per axis". Pass --mode grid to run the
full Cartesian product instead.

How config reaches the worker processes: pipeline.py reads module-level globals
(grid_w, model_name, VIEWS, ...). Process pools on Windows use 'spawn', which
re-imports pipeline.py with its DEFAULT globals, so per-config overrides set in
the parent would be lost. We therefore pass the config to every pool via an
`initializer` that calls apply_config() inside each worker. The segmentation
threshold is injected by monkeypatching pipeline.segment_largest_component (the
patched closure is built inside the worker, never pickled). pipeline.py, the
notebook and run_parallel.py are left untouched.

Timing: the three passes mirror run_parallel.py
  Pass 1  segment + render (+ head projection)
  Pass 2  DINO tokens + prototype + top-k   (model load time recorded separately)
  Pass 3  triangulate + rotate + QA render
Each pass is timed with perf_counter and written to summary.csv.

Correctness is checked by hand. For every individual the rotated volume is
re-rendered and stitched into one image under results/<config>/composites/, so
each config folder can be eyeballed and the errors counted manually. The QA
composite ALWAYS uses the 6 canonical views -- even for the 4-view configs -- so
every config produces a comparable 6-panel image.

Run (repo root):
  python run_experiments.py                      # full OFAT sweep, animals=[GH]
  python run_experiments.py --animals GH MA      # more species
  python run_experiments.py --only baseline model=large grid=90
  python run_experiments.py --keep-work          # keep heavy intermediates

On the DTU HPC cluster, submit via run_experiments_lsf.sh (bsub) or
run_experiments_slurm.sh (sbatch); both request a CUDA GPU for Pass 2.
"""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import json
import math
import multiprocessing
import os
import shutil
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
from scipy import ndimage
import vedo
import tifffile

import pipeline as P
from pipeline import (
    find_input_volumes, process_volume, get_head_information,
    load_dino_model, extract_patch_tokens,
    discover_segmented_individuals, save_top_k_patches,
    build_rays_from_individual, ransac_fuse,
    rotation_matrix_from_vectors, rotate_volume,
)

# Canonical 6 views, captured before any apply_config() mutates pipeline.VIEWS.
VIEWS_6 = list(P.VIEWS)


def _pyramid_up(direction: np.ndarray) -> np.ndarray:
    """Up vector for a pyramid camera: world +Z, with a +Y fallback if the
    viewing direction runs nearly parallel to +Z (which would make +Z a
    degenerate up vector)."""
    d = direction / np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(d, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    return up


# "4 views": cameras at the vertices of a regular tetrahedron, all aimed at the
# volume center. Tetrahedron vertices are maximally spread in 3D and -- unlike a
# symmetric square pyramid, whose four origins share one height and are coplanar
# -- non-coplanar, so the triangulated head depth stays well-conditioned.
_PYRAMID_DIRS = [
    ("P1", np.array([ 1.0,  1.0,  1.0])),
    ("P2", np.array([ 1.0, -1.0, -1.0])),
    ("P3", np.array([-1.0,  1.0, -1.0])),
    ("P4", np.array([-1.0, -1.0,  1.0])),
]
VIEWS_4 = [
    (label, d / np.linalg.norm(d), _pyramid_up(d)) for label, d in _PYRAMID_DIRS
]

MODEL_IDS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base":  "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "huge":  "facebook/dinov3-vith16plus-pretrain-lvd1689m",  # gated; downloads on first use
}

# Segmentation threshold. "otsu" = skimage's automatic threshold (None below).
# 96/97/98 are PERCENTILES of the volume intensity: voxels below that percentile
# count as background, voxels above it are foreground (higher percentile => less
# foreground kept). None => otsu (the pipeline default).
THRESHOLDS = {"otsu": None, "96": 96, "97": 97, "98": 98}

# Force 'spawn' for the pools: Pass 3 is created after Pass 2 initialises CUDA in
# the main process, and forking after a live CUDA context is fragile on Linux.
# spawn re-imports the module and runs the initializer cleanly (the config-via-
# initializer design already assumes spawn, as on Windows).
_SPAWN = multiprocessing.get_context("spawn")


# --------------------------------------------------------------------------- #
# Config application (runs in the main process AND in every pool worker)
# --------------------------------------------------------------------------- #
def apply_config(cfg: dict) -> None:
    """Push one experiment config into pipeline.py's module globals.

    Called once in the main process per config, and again inside each pool
    worker via the pool initializer (spawn re-imports pipeline with defaults).
    """
    P.VIEWS = cfg["views"]
    P.model_name = cfg["model_name"]

    g = int(cfg["grid"])
    P.grid_w = g
    P.grid_h = g
    P.image_w = g * P.patch_size
    P.image_h = g * P.patch_size
    P.render_size = (P.image_w, P.image_h)

    thr = cfg["seg_threshold"]  # None -> otsu, else a percentile (e.g. 97)

    def _segment(volume, _thr=thr):
        t = threshold_otsu(volume) if _thr is None else np.percentile(volume, _thr)
        mask = volume > t
        mask = ndimage.binary_dilation(mask, iterations=5)
        mask = ndimage.binary_fill_holes(mask)
        labeled, num = ndimage.label(mask)
        if num == 0:
            return mask
        sizes = ndimage.sum_labels(volume, labeled, index=np.arange(1, num + 1))
        largest = int(np.argmax(sizes) + 1)
        return labeled == largest

    # process_volume() calls the bare name segment_largest_component, resolved
    # from pipeline's module globals at call time, so this swap takes effect.
    P.segment_largest_component = _segment


def _worker_init(cfg: dict) -> None:
    apply_config(cfg)


# --------------------------------------------------------------------------- #
# Pool worker functions (top-level so 'spawn' can import them by reference)
# --------------------------------------------------------------------------- #
def _segment_worker(in_path: Path, in_root: Path, out_root: Path) -> str:
    process_volume(in_path, in_root, out_root)
    return f"{in_path.parts[-2]}/{in_path.stem}"


def _rotate_worker(animal: str, individual: str, original_path: Path, info_path: Path) -> dict:
    """Pass 3 unit: fuse head rays, rotate to the largest axis, QA-render.

    The QA re-render always uses the 6 canonical views (VIEWS_6) regardless of
    how many views the config used upstream, so every composite has 6 panels and
    the 4-view configs stay visually comparable to the rest.
    """
    output_root = info_path / "segmented"
    tokens_root = info_path / "tokens"
    res = {"animal": animal, "individual": individual, "status": "ok",
           "n_inliers": 0, "n_rays": 0}

    origins, dirs = build_rays_from_individual(output_root, tokens_root, animal, individual)
    res["n_rays"] = len(origins)
    if len(origins) < 2:
        res["status"] = "skip_no_rays"
        return res

    fused = ransac_fuse(origins, dirs, threshold=P.ransac_threshold, refine_iters=P.ransac_iterations)
    if fused is None:
        res["status"] = "skip_ransac"
        return res

    head_xyz = fused["point"]
    meta = json.loads((output_root / animal / individual / "metadata.json").read_text(encoding="utf-8"))
    com_xyz = np.array(meta["center_of_mass_xyz"], dtype=float)
    res["n_inliers"] = int(fused["n_inliers"])

    volume = vedo.load(str(original_path / animal / f"{individual}.tif")).tonumpy()
    target_axis = ["+X", "+Y", "+Z"][int(np.argmax(meta["volume_shape"]))]
    rot = rotation_matrix_from_vectors(head_xyz - com_xyz, P.AXIS_TO_VECTOR[target_axis])
    rotated = rotate_volume(volume, rot=rot, order=1, center=com_xyz)

    out_tif_dir = info_path / "final" / animal
    out_tif_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_tif_dir / f"{individual}.tif"
    tifffile.imwrite(str(out_tif), np.transpose(rotated, (2, 1, 0)).astype(volume.dtype, copy=False))

    # QA re-render of the rotated volume -> these PNGs become the composite.
    # Force the full 6 views so a 4-view config still yields a 6-panel image.
    # Pool workers are REUSED across individuals, so restore the config's VIEWS
    # afterwards: otherwise the next individual on this worker would run
    # build_rays_from_individual with VIEWS_6, and for the 4-view config (whose
    # metadata labels are P1..P4) every angle lookup would miss -> zero rays ->
    # skip_no_rays. This is why ~workers-worth of 4-view individuals passed and
    # the rest silently failed.
    cfg_views = P.VIEWS
    try:
        P.VIEWS = VIEWS_6
        process_volume(out_tif, info_path / "final", info_path / "final_segmented")
    finally:
        P.VIEWS = cfg_views
    res["target_axis"] = target_axis
    return res


# --------------------------------------------------------------------------- #
# Prototype with a limited number of annotation sources
# --------------------------------------------------------------------------- #
def build_prototype_limited(animal, tokens_root, segmented_root, individuals, single_image: bool):
    """Average the ground-truth head patch token over a chosen set of sources.

    individuals   : individual stems to draw from (already trimmed to 1 or 5)
    single_image  : if True, use only the first view of the (single) individual
    """
    proto = []
    for ind in individuals:
        head_info_path = segmented_root / animal / ind / "head_projection.json"
        if not head_info_path.exists():
            continue
        head_info = json.loads(head_info_path.read_text(encoding="utf-8"))
        items = list(head_info.items())
        if single_image:
            items = items[:1]
        for angle, info in items:
            tokens_path = tokens_root / animal / ind / f"{ind}_{angle}" / "tokens.npy"
            if not tokens_path.exists():
                continue
            tokens = np.load(tokens_path, mmap_mode="r")
            idx = int(info["patch_row"]) * P.grid_w + int(info["patch_col"])
            proto.append(tokens[idx].copy())
    if not proto:
        raise ValueError(f"No prototype vectors for '{animal}' with the chosen annotation set")
    pv = np.mean(proto, axis=0)
    np.save(tokens_root / animal / "prototype_vector.npy", pv)
    return pv


# --------------------------------------------------------------------------- #
# Helpers: composite montage
# --------------------------------------------------------------------------- #
def make_composite(png_paths: list[Path], labels: list[str], out_path: Path,
                   tile: int = 320, header: str = "") -> bool:
    imgs = []
    for p in png_paths:
        if p.exists():
            imgs.append((Image.open(p).convert("RGB").resize((tile, tile)), ""))
    if not imgs:
        return False
    cols = 3 if len(imgs) > 4 else 2
    rows = math.ceil(len(imgs) / cols)
    pad = 24 if header else 0
    canvas = Image.new("RGB", (cols * tile, rows * tile + pad), "white")
    draw = ImageDraw.Draw(canvas)
    if header:
        draw.text((6, 6), header, fill="black")
    for i, ((im, _), lab) in enumerate(zip(imgs, labels + [""] * len(imgs))):
        x = (i % cols) * tile
        y = (i // cols) * tile + pad
        canvas.paste(im, (x, y))
        draw.rectangle([x + 4, y + 4, x + 64, y + 22], fill="white")
        draw.text((x + 8, y + 6), lab, fill="red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


def make_overlay_composite(items: list, out_path: Path, tile: int = 320, header: str = "") -> bool:
    """Like make_composite, but draws the projected head marker on each view.

    items: list of (png_path, angle_label, (px, py) | None) in full-resolution
    rendered-image pixels. A red marker means the point is inside the view; an
    orange marker pinned to the edge means it projected off-view (e.g. the head
    is occluded / behind the camera in that direction).
    """
    tiles = []
    for png, lab, pt in items:
        if not png.exists():
            continue
        im = Image.open(png).convert("RGB")
        if pt is not None:
            d = ImageDraw.Draw(im)
            px, py = pt
            r = max(8, int(im.width * 0.025))
            inb = 0 <= px < im.width and 0 <= py < im.height
            color = "red" if inb else "orange"
            dx = min(max(px, r), im.width - r)
            dy = min(max(py, r), im.height - r)
            d.ellipse([dx - r, dy - r, dx + r, dy + r], outline=color, width=max(3, r // 3))
            d.line([dx - r, dy, dx + r, dy], fill=color, width=2)
            d.line([dx, dy - r, dx, dy + r], fill=color, width=2)
            if not inb:
                lab += " (off-view)"
        tiles.append((im.resize((tile, tile)), lab))
    if not tiles:
        return False
    cols = 3 if len(tiles) > 4 else 2
    rows = math.ceil(len(tiles) / cols)
    pad = 24 if header else 0
    canvas = Image.new("RGB", (cols * tile, rows * tile + pad), "white")
    draw = ImageDraw.Draw(canvas)
    if header:
        draw.text((6, 6), header, fill="black")
    for i, (im, lab) in enumerate(tiles):
        x = (i % cols) * tile
        y = (i // cols) * tile + pad
        canvas.paste(im, (x, y))
        draw.rectangle([x + 4, y + 4, x + 120, y + 22], fill="white")
        draw.text((x + 8, y + 6), lab, fill="blue")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


# --------------------------------------------------------------------------- #
# Annotation verification (sanity-check the 3-D head point before sweeping)
# --------------------------------------------------------------------------- #
def verify_annotations(original_path: Path, results_root: Path, annotations: dict,
                       animals: list[str], n_workers: int) -> None:
    """Render the annotated individuals and overlay the projected head point.

    Uses 6 views at grid 60 (no DINO model needed -- this only checks the
    projection geometry). The red circle should sit on the head in each view
    where the head is visible; that confirms the [x, y, z] annotation is right.
    """
    cfg = {"views": VIEWS_6, "model_name": MODEL_IDS["base"], "grid": 60, "seg_threshold": None}
    apply_config(cfg)
    out_dir = results_root / "verify_annotations"
    seg_root = out_dir / "segmented"
    out_dir.mkdir(parents=True, exist_ok=True)

    vols = []
    for a in animals:
        for fname in sorted(annotations.get(a, {})):
            f = original_path / a / fname
            if f.exists():
                vols.append(f)
            else:
                print(f"   missing volume (annotated but not on disk): {f}")
    if not vols:
        print(f"No annotated volumes found for {animals}")
        return

    print(f"verify-annotations: rendering {len(vols)} annotated volumes (6 views, grid 60)")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_SPAWN,
                             initializer=_worker_init, initargs=(cfg,)) as ex:
        futs = [ex.submit(_segment_worker, f, original_path, seg_root) for f in vols]
        for fut in as_completed(futs):
            fut.result()

    for f in vols:
        a, ind = f.parts[-2], f.stem
        anno = annotations[a][f.name]
        get_head_information(f, original_path, seg_root, anno)
        hp = json.loads((seg_root / a / ind / "head_projection.json").read_text(encoding="utf-8"))
        items = []
        for lab, *_ in VIEWS_6:
            png = seg_root / a / ind / f"{ind}_{lab}.png"
            info = hp.get(lab)
            pt = None
            if info is not None:
                pt = ((info["patch_col"] + 0.5) * P.patch_size,
                      (info["patch_row"] + 0.5) * P.patch_size)
            items.append((png, lab, pt))
        make_overlay_composite(items, out_dir / f"{a}__{ind}.png", header=f"{a}/{ind} | head xyz={anno}")
        print(f"   {a}/{ind} -> overlay")

    shutil.rmtree(seg_root, ignore_errors=True)  # keep only the overlays
    print(f"\nOverlays under {out_dir}/")
    print("Red circle = projected head. Check it lands on the head in the views "
          "where the head is visible (orange = projected off-view).")


# --------------------------------------------------------------------------- #
# One config end-to-end
# --------------------------------------------------------------------------- #
def individual_index(stem: str):
    """The individual number = the LAST underscore token of the file stem.

    Robust to the group token being 1 or 10: gras_1_015 -> 15, mel_10_015 -> 15.
    Returns None if the stem has no trailing integer.
    """
    tok = stem.rsplit("_", 1)[-1]
    return int(tok) if tok.isdigit() else None


def run_config(params: dict, cfg: dict, original_path: Path, results_root: Path,
               annotations: dict, animals: list[str], n_workers: int,
               keep_work: bool, eval_range: tuple, all_vols: list) -> dict:
    cfg_id = params["id"]
    cfg_dir = results_root / cfg_id
    work = cfg_dir / "work"
    comp_dir = cfg_dir / "composites"
    work.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    apply_config(cfg)  # main process (Pass 1b + Pass 2 run here)
    output_root = work / "segmented"
    tokens_root = work / "tokens"

    # Evaluation set. Files are named <prefix>_<group>_<NNN>.tif and the dataset
    # has many scan groups (gras_1_*, gras_17_*, ...). An individual is in the
    # eval set only if it (a) shares a prefix with an annotated individual -- so
    # it belongs to the prototype's id family, e.g. gras_1_* -- and (b) its
    # trailing index _NNN is in eval_range (default 11-20). The prefix match is
    # essential: without it every group's _011.._020 would match and hundreds of
    # unrelated individuals would be processed.
    lo, hi = eval_range

    def _prefix(stem):
        return stem.rsplit("_", 1)[0]  # gras_1_015 -> "gras_1", mel_10_002 -> "mel_10"

    anno_prefixes = {a: {_prefix(Path(fn).stem) for fn in annotations.get(a, {})} for a in animals}
    eval_volumes = []
    for f in all_vols:
        idx = individual_index(f.stem)
        if idx is None or not (lo <= idx <= hi):
            continue
        if _prefix(f.stem) in anno_prefixes.get(f.parts[-2], set()):
            eval_volumes.append(f)

    # Prototype sources: the annotated individuals (image_annotations.json). They
    # may sit outside eval_range, but must be rendered + tokenized so the head
    # prototype can be built. They are not rotated/composited themselves.
    proto_volumes = []
    for a in animals:
        for fname in annotations.get(a, {}):
            f = original_path / a / fname
            if f.exists():
                proto_volumes.append(f)

    # Render + tokenize the union; rotate/composite only the evaluation set.
    volumes = sorted(set(eval_volumes) | set(proto_volumes))
    if not volumes:
        raise RuntimeError("no input volumes found (eval set and prototype sources are both empty)")
    if not eval_volumes:
        print(f"   [{cfg_id}] WARNING: no individuals with index in {lo}-{hi}; no composites produced")
    eval_keys = {(f.parts[-2], f.stem) for f in eval_volumes}

    qa_labels = [v[0] for v in VIEWS_6]  # composite is always the 6 canonical views
    timings = {}

    # --- Pass 1: segment + render (parallel) + head projection (serial) ------
    print(f"   Pass 1/3: segment + render {len(volumes)} volumes "
          f"({len(eval_volumes)} eval + {len(proto_volumes)} annotated) on {n_workers} workers",
          flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_SPAWN,
                             initializer=_worker_init, initargs=(cfg,)) as ex:
        futs = [ex.submit(_segment_worker, f, original_path, output_root) for f in volumes]
        for fut in as_completed(futs):
            fut.result()
    for f in volumes:
        anno = annotations.get(f.parts[-2], {}).get(f.name)
        if anno is not None:
            get_head_information(f, original_path, output_root, anno)
    timings["pass1_s"] = time.perf_counter() - t0
    print(f"   Pass 1 done in {timings['pass1_s']:.0f}s", flush=True)

    # --- Pass 2: DINO tokens + prototype + top-k (serial; GPU/all cores) -----
    print(f"   Pass 2/3: DINO tokens for {len(volumes)} volumes x {len(cfg['views'])} views "
          f"(model={params['model']})", flush=True)
    t0 = time.perf_counter()
    t_load = time.perf_counter()
    processor, model, device = load_dino_model()
    load_s = time.perf_counter() - t_load
    for f in volumes:
        animal, individual = f.parts[-2], f.stem
        for angle, *_ in cfg["views"]:
            img = output_root / animal / individual / f"{individual}_{angle}.png"
            extract_patch_tokens(img, output_root, tokens_root, processor, model, device)

    anno_mode = params["anno_mode"]
    for animal in animals:
        annotated = [Path(fn).stem for fn in sorted(annotations.get(animal, {}))]
        if anno_mode == "1_image":
            sources, single = annotated[:1], True
        elif anno_mode == "1_animal":
            sources, single = annotated[:1], False
        else:  # "5_animals"
            sources, single = annotated[:5], False
        try:
            build_prototype_limited(animal, tokens_root, output_root, sources, single)
        except ValueError as e:
            print(f"   [{cfg_id}] {e}")
            continue
        save_top_k_patches(tokens_root, output_root, animal, k=int(params["top_k"]))
    timings["pass2_s"] = time.perf_counter() - t0
    timings["load_s"] = load_s
    print(f"   Pass 2 done in {timings['pass2_s']:.0f}s (model load {load_s:.0f}s)", flush=True)

    # free the model before the next config (esp. large/huge on a shared GPU)
    del model, processor
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # --- Pass 3: triangulate + rotate + QA render (parallel) -----------------
    jobs = [(a, ind) for a in animals
            for ind in discover_segmented_individuals(output_root, a) if (a, ind) in eval_keys]
    print(f"   Pass 3/3: rotate + QA render {len(jobs)} individuals on {n_workers} workers", flush=True)
    t0 = time.perf_counter()
    rot_results = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_SPAWN,
                             initializer=_worker_init, initargs=(cfg,)) as ex:
        futs = [ex.submit(_rotate_worker, a, ind, original_path, work) for a, ind in jobs]
        for fut in as_completed(futs):
            rot_results.append(fut.result())
    timings["pass3_s"] = time.perf_counter() - t0
    print(f"   Pass 3 done in {timings['pass3_s']:.0f}s", flush=True)

    # --- Build composites (manual error check) -------------------------------
    per_individual = []
    n_ok = 0
    for r in rot_results:
        animal, individual = r["animal"], r["individual"]
        if r["status"] == "ok":
            n_ok += 1
        # composite from the 6-view QA render of the rotated volume
        qa_dir = work / "final_segmented" / animal / individual
        png_paths = [qa_dir / f"{individual}_{lab}.png" for lab in qa_labels]
        header = f"{cfg_id} | {animal}/{individual}"
        if r["status"] != "ok":
            header += f" | {r['status']}"
        make_composite(png_paths, qa_labels, comp_dir / f"{animal}__{individual}.png", header=header)
        per_individual.append({
            "animal": animal, "individual": individual, "status": r["status"],
            "n_inliers": r["n_inliers"], "n_rays": r["n_rays"],
        })

    metrics = {
        "id": cfg_id,
        "params": params,
        "timings": timings,
        "n_individuals": len(rot_results),
        "n_ok": n_ok,
        "per_individual": per_individual,
    }
    (cfg_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)

    return metrics


# --------------------------------------------------------------------------- #
# Config grid (OFAT)
# --------------------------------------------------------------------------- #
BASELINE = {
    "views": "6", "model": "base", "grid": 60,
    "threshold": "otsu", "top_k": 5, "anno_mode": "5_animals",
}

SWEEPS = {
    "views":     ["6", "4"],
    "model":     ["small", "base", "large", "huge"],
    "grid":      [30, 60, 90],
    "threshold": ["otsu", "96", "97", "98"],
    "top_k":     [3, 5, 8],
    "anno_mode": ["1_image", "1_animal", "5_animals"],
}


def _materialize(spec: dict) -> tuple[dict, dict]:
    """Turn a human-readable spec into (params, runtime cfg)."""
    params = {
        "id": spec["id"],
        "axis": spec["axis"],
        "level": spec["level"],
        "views": spec["views"],
        "model": spec["model"],
        "grid": spec["grid"],
        "threshold": spec["threshold"],
        "top_k": spec["top_k"],
        "anno_mode": spec["anno_mode"],
    }
    cfg = {
        "views": VIEWS_6 if spec["views"] == "6" else VIEWS_4,
        "model_name": MODEL_IDS[spec["model"]],
        "grid": spec["grid"],
        "seg_threshold": THRESHOLDS[spec["threshold"]],
    }
    return params, cfg


def build_ofat_configs() -> list[dict]:
    """Baseline + one varied axis at a time (deduplicated against baseline)."""
    configs = []
    base = dict(BASELINE, id="baseline", axis="baseline", level="baseline")
    configs.append(base)
    for axis, levels in SWEEPS.items():
        for lvl in levels:
            if str(lvl) == str(BASELINE[axis]):
                continue  # baseline level already covered
            spec = dict(BASELINE)
            spec[axis] = lvl
            spec["id"] = f"{axis}={lvl}"
            spec["axis"] = axis
            spec["level"] = str(lvl)
            configs.append(spec)
    return configs


def build_grid_configs() -> list[dict]:
    """Full Cartesian product (2592 configs). Off by default."""
    import itertools
    keys = list(SWEEPS)
    configs = []
    for combo in itertools.product(*(SWEEPS[k] for k in keys)):
        spec = dict(zip(keys, combo))
        spec["id"] = "_".join(f"{k}={spec[k]}" for k in keys)
        spec["axis"] = "grid"
        spec["level"] = spec["id"]
        configs.append(spec)
    return configs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
SUMMARY_FIELDS = [
    "id", "axis", "level", "views", "model", "grid", "threshold", "top_k", "anno_mode",
    "n_individuals", "n_ok",
    "pass1_s", "pass2_s", "pass3_s", "load_s", "total_s", "status",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default="data/original_photos", help="root with <SPECIES>/<ind>.tif")
    ap.add_argument("--results", default="data/experiments", help="output root")
    ap.add_argument("--animals", nargs="+", default=["GH"], help="species codes to evaluate")
    ap.add_argument("--workers", type=int, default=0, help="0 => SLURM_CPUS_PER_TASK or cpu_count-2")
    ap.add_argument("--mode", choices=["ofat", "grid"], default="ofat")
    ap.add_argument("--eval-range", nargs=2, type=int, default=[11, 20], metavar=("LO", "HI"),
                    help="individual index range (trailing _NNN) to evaluate/composite")
    ap.add_argument("--only", nargs="+", default=None, help="run only configs whose id is listed")
    ap.add_argument("--keep-work", action="store_true", help="keep heavy intermediates (segmented/tokens/final)")
    ap.add_argument("--verify-annotations", action="store_true",
                    help="render the annotated individuals, overlay the projected head, then exit")
    args = ap.parse_args()

    original_path = Path(args.original)
    results_root = Path(args.results)
    results_root.mkdir(parents=True, exist_ok=True)

    if args.workers > 0:
        n_workers = args.workers
    else:
        # LSF (DTU central HPC) sets LSB_DJOB_NUMPROC; SLURM (Sophia) sets
        # SLURM_CPUS_PER_TASK. Fall back to local cores minus headroom.
        env_cpus = os.environ.get("LSB_DJOB_NUMPROC") or os.environ.get("SLURM_CPUS_PER_TASK")
        n_workers = int(env_cpus) if env_cpus else max(1, (os.cpu_count() or 4) - 2)

    annotations = json.loads(
        Path("Annoteringer/annotations_output/image_annotations.json").read_text(encoding="utf-8")
    )

    if args.verify_annotations:
        verify_annotations(original_path, results_root, annotations, args.animals, n_workers)
        return

    # Discover the input volumes once (the dataset can hold thousands of files on
    # a network filesystem; this avoids re-listing it for every config).
    print("Discovering input volumes ...", flush=True)
    all_vols = [f for f in find_input_volumes(original_path) if f.parts[-2] in args.animals]
    print(f"Found {len(all_vols)} volumes for {args.animals}", flush=True)

    configs = build_grid_configs() if args.mode == "grid" else build_ofat_configs()
    if args.only:
        wanted = set(args.only)
        configs = [c for c in configs if c["id"] in wanted]

    print(f"Experiments: {len(configs)} configs | animals={args.animals} | workers={n_workers}")
    print(f"Results -> {results_root}\n")

    summary_path = results_root / "summary.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()

        for i, spec in enumerate(configs, 1):
            params, cfg = _materialize(spec)
            cfg_id = params["id"]
            print(f"[{i}/{len(configs)}] {cfg_id} ...", flush=True)
            row = {k: params.get(k, "") for k in SUMMARY_FIELDS}
            t_cfg = time.perf_counter()
            try:
                m = run_config(params, cfg, original_path, results_root, annotations,
                               args.animals, n_workers, args.keep_work, tuple(args.eval_range), all_vols)
                row.update({
                    "n_individuals": m["n_individuals"], "n_ok": m["n_ok"],
                    "pass1_s": round(m["timings"]["pass1_s"], 2),
                    "pass2_s": round(m["timings"]["pass2_s"], 2),
                    "pass3_s": round(m["timings"]["pass3_s"], 2),
                    "load_s": round(m["timings"]["load_s"], 2),
                    "total_s": round(time.perf_counter() - t_cfg, 2),
                    "status": "ok",
                })
                print(f"      done: {m['n_ok']}/{m['n_individuals']} rotated | "
                      f"p1={row['pass1_s']}s p2={row['pass2_s']}s p3={row['pass3_s']}s | "
                      f"check composites under {results_root}/{cfg_id}/composites/")
            except Exception as e:
                row["status"] = f"FAILED: {e}"
                row["total_s"] = round(time.perf_counter() - t_cfg, 2)
                print(f"      FAILED: {e}")
                traceback.print_exc()
            writer.writerow(row)
            fh.flush()

    print(f"\nSummary written to {summary_path}")
    print("Composites for manual correctness counting are under "
          f"{results_root}/<config_id>/composites/")


if __name__ == "__main__":
    main()
