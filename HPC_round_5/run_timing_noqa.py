"""
HPC round 5 -- pipeline TIMING only, WITHOUT the QA re-render.

Goal: re-measure the per-pass timings of the ACTUAL pipeline (both the DINO sweep
and the PCA method) with the final QA re-render REMOVED. The QA re-render (a 6-view
re-render of the already-rotated volume) is only a manual-correctness aid -- it is
not part of producing the reoriented volume -- yet in the earlier rounds it
dominated Pass 3 (and, for PCA, was the only rendering at all). Dropping it gives a
clean measurement of the real reorientation cost, and makes the PCA run finish in
seconds.

What is and is NOT removed
--------------------------
* DINO: the QA re-render in Pass 3 is removed. The Pass 1 render of the 6 views is
  KEPT -- it is REQUIRED (DINO extracts patch tokens from those images; it is part
  of the method, not a QA aid). So DINO here = segment+render (Pass 1) + tokens
  (Pass 2) + triangulate+rotate+write (Pass 3, no QA render).
* PCA: the QA re-render in Pass 3 is removed. PCA needs no rendering otherwise, so
  PCA here = segment (Pass 1) + PCA principal axis (Pass 2) + rotate+write (Pass 3).

No composites are produced (there is nothing to eyeball); only timings + n_ok land
in summary.csv / metrics.json. Correctness was already established in rounds 1-4.

Reuse / no edits to the base code
---------------------------------
This wrapper imports pipeline.py and run_experiments.py and edits NEITHER. It reuses
run_experiments' config grid, eval-set selection helpers, segmentation/token/
prototype passes and constants verbatim; it only re-implements the two things that
must change: a Pass-3 rotate worker with the QA render stripped, and the per-config
driver that calls it and skips composite building. The DINO Pass-1/Pass-3 pools use
run_experiments' own initializer (_worker_init -> apply_config) so per-config globals
(views/grid/threshold) reach the workers exactly as before.

Why the workers live at module level: the process pools use 'spawn', which re-imports
this script in every worker; top-level worker functions (guarded by the __main__
block) are therefore importable/picklable in the children, same pattern as
run_experiments.py.

summary.csv uses the SAME columns as run_experiments.py (imported verbatim), so the
DINO rows and the PCA row stack together and line up with the earlier rounds.

Run (repo root, same env/venv as the other rounds)
--------------------------------------------------
  python HPC_round_5/run_timing_noqa.py --animals GH \
      --original data/original_photos \
      --results  HPC_round_5/experiments

  # subset / control:
  --methods dino            # only the DINO sweep (default: dino pca -> both)
  --methods pca             # only PCA
  --only baseline grid=90   # only these DINO configs
  --mode ofat|grid          # DINO config set (default ofat = 14 configs)

On the DTU HPC cluster, submit via HPC_round_5/run_timing_noqa_lsf.sh (bsub). It
requests the gpuv100 queue + a GPU (needed for DINO Pass 2) and wraps the run in
xvfb-run (still needed for DINO's Pass 1 render). PCA uses neither, but shares the
job so its Pass 3 is timed on the same node as everything else.
"""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import json
import multiprocessing
import os
import shutil
import sys
import time
import traceback

import numpy as np
import tifffile
import vedo

# repo root = parent of this HPC_round_5/ folder
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline as P  # noqa: E402
from pipeline import (  # noqa: E402
    find_input_volumes, get_head_information,
    load_dino_model, extract_patch_tokens,
    discover_segmented_individuals, save_top_k_patches,
    build_rays_from_individual, ransac_fuse,
    rotation_matrix_from_vectors, rotate_volume,
)
import run_experiments as RE  # noqa: E402

_SPAWN = multiprocessing.get_context("spawn")


# =========================================================================== #
# DINO: Pass-3 rotate worker WITHOUT the QA re-render
# =========================================================================== #
def _rotate_worker_noqa(animal: str, individual: str, original_path: Path, info_path: Path) -> dict:
    """Copy of run_experiments._rotate_worker with the QA re-render removed.

    Does the real Pass-3 work -- fuse head rays, rotate the volume onto the largest
    axis, write the rotated .tif -- but NOT the 6-view QA re-render of the result.
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
    # (QA re-render removed -- this is the only difference vs run_experiments._rotate_worker)
    res["target_axis"] = target_axis
    return res


def run_dino_config_timing(params: dict, cfg: dict, original_path: Path, results_root: Path,
                           annotations: dict, animals: list, n_workers: int, keep_work: bool,
                           eval_range: tuple, all_vols: list) -> dict:
    """run_experiments.run_config, minus the QA render and composite building.

    Passes 1 and 2 are reused verbatim (segmentation + head projection, then DINO
    tokens + prototype + top-k). Pass 3 uses _rotate_worker_noqa above.
    """
    cfg_id = params["id"]
    cfg_dir = results_root / cfg_id
    work = cfg_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    RE.apply_config(cfg)  # main process
    output_root = work / "segmented"
    tokens_root = work / "tokens"

    # --- eval-set selection (identical to run_experiments.run_config) ---------
    lo, hi = eval_range

    def _prefix(stem):
        return stem.rsplit("_", 1)[0]

    anno_prefixes = {a: {_prefix(Path(fn).stem) for fn in annotations.get(a, {})} for a in animals}
    eval_volumes = []
    for f in all_vols:
        idx = RE.individual_index(f.stem)
        if idx is None or not (lo <= idx <= hi):
            continue
        if _prefix(f.stem) in anno_prefixes.get(f.parts[-2], set()):
            eval_volumes.append(f)

    proto_volumes = []
    for a in animals:
        for fname in annotations.get(a, {}):
            f = original_path / a / fname
            if f.exists():
                proto_volumes.append(f)

    volumes = sorted(set(eval_volumes) | set(proto_volumes))
    if not volumes:
        raise RuntimeError("no input volumes found (eval set and prototype sources are both empty)")
    if not eval_volumes:
        print(f"   [{cfg_id}] WARNING: no individuals with index in {lo}-{hi}; nothing to rotate")
    eval_keys = {(f.parts[-2], f.stem) for f in eval_volumes}
    timings = {}

    # --- Pass 1: segment + render (parallel) + head projection (serial) ------
    print(f"   Pass 1/3: segment + render {len(volumes)} volumes "
          f"({len(eval_volumes)} eval + {len(proto_volumes)} annotated) on {n_workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_SPAWN,
                             initializer=RE._worker_init, initargs=(cfg,)) as ex:
        futs = [ex.submit(RE._segment_worker, f, original_path, output_root) for f in volumes]
        for fut in as_completed(futs):
            fut.result()
    for f in volumes:
        anno = annotations.get(f.parts[-2], {}).get(f.name)
        if anno is not None:
            get_head_information(f, original_path, output_root, anno)
    timings["pass1_s"] = time.perf_counter() - t0
    print(f"   Pass 1 done in {timings['pass1_s']:.0f}s", flush=True)

    # --- Pass 2: DINO tokens + prototype + top-k (serial) --------------------
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
            RE.build_prototype_limited(animal, tokens_root, output_root, sources, single)
        except ValueError as e:
            print(f"   [{cfg_id}] {e}")
            continue
        save_top_k_patches(tokens_root, output_root, animal, k=int(params["top_k"]))
    timings["pass2_s"] = time.perf_counter() - t0
    timings["load_s"] = load_s
    print(f"   Pass 2 done in {timings['pass2_s']:.0f}s (model load {load_s:.0f}s)", flush=True)

    del model, processor
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # --- Pass 3: triangulate + rotate + write (parallel, NO QA render) -------
    jobs = [(a, ind) for a in animals
            for ind in discover_segmented_individuals(output_root, a) if (a, ind) in eval_keys]
    print(f"   Pass 3/3: triangulate + rotate {len(jobs)} individuals on {n_workers} workers "
          f"(no QA render)", flush=True)
    t0 = time.perf_counter()
    rot_results = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_SPAWN,
                             initializer=RE._worker_init, initargs=(cfg,)) as ex:
        futs = [ex.submit(_rotate_worker_noqa, a, ind, original_path, work) for a, ind in jobs]
        for fut in as_completed(futs):
            rot_results.append(fut.result())
    timings["pass3_s"] = time.perf_counter() - t0
    print(f"   Pass 3 done in {timings['pass3_s']:.0f}s", flush=True)

    n_ok = sum(1 for r in rot_results if r["status"] == "ok")
    metrics = {
        "id": cfg_id, "params": params, "timings": timings,
        "n_individuals": len(rot_results), "n_ok": n_ok,
        "note": "TIMING ONLY, QA re-render removed. Pass 1 render kept (DINO needs it).",
        "per_individual": [{"animal": r["animal"], "individual": r["individual"],
                            "status": r["status"], "n_inliers": r["n_inliers"],
                            "n_rays": r["n_rays"]} for r in rot_results],
    }
    (cfg_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return metrics


# =========================================================================== #
# PCA: same as HPC_round_4, but Pass 3 does NOT QA-render
# =========================================================================== #
PCA_ID = "pca_seg"


def _pca_segment_worker(in_path: Path) -> dict:
    """Pass 1 (PCA): load + segment; return foreground voxel coordinates."""
    animal, individual = in_path.parts[-2], in_path.stem
    res = {"animal": animal, "individual": individual, "status": "ok",
           "n_voxels": 0, "shape": None, "coords": None, "seg_s": 0.0}
    t = time.perf_counter()
    volume = vedo.load(str(in_path)).tonumpy()
    mask = P.segment_largest_component(volume)
    res["seg_s"] = time.perf_counter() - t
    n = int(mask.sum())
    res["n_voxels"] = n
    res["shape"] = [int(s) for s in volume.shape]
    if n < 10:
        res["status"] = "skip_empty_mask"
        return res
    res["coords"] = np.argwhere(mask).astype(np.int32)
    return res


def _pca_principal_axis(coords: np.ndarray):
    coords = coords.astype(np.float64)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = (centered.T @ centered) / coords.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    axis = evecs[:, 0]
    if axis[int(np.argmax(np.abs(axis)))] < 0.0:
        axis = -axis
    return axis, centroid, evals[0], float(evals[0] / max(evals[1], 1e-12))


def _pca_rotate_worker(in_path: Path, work: Path, principal_axis: list,
                       centroid: list, target_axis: str) -> dict:
    """Pass 3 (PCA): Rodrigues rotate + write .tif. NO QA re-render."""
    animal, individual = in_path.parts[-2], in_path.stem
    res = {"animal": animal, "individual": individual, "rotate_s": 0.0}
    volume = vedo.load(str(in_path)).tonumpy()
    t = time.perf_counter()
    rot = rotation_matrix_from_vectors(np.asarray(principal_axis, dtype=float),
                                       P.AXIS_TO_VECTOR[target_axis])
    rotated = rotate_volume(volume, rot=rot, order=1, center=np.asarray(centroid, dtype=float))
    res["rotate_s"] = time.perf_counter() - t
    out_dir = work / "final" / animal
    out_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_dir / f"{individual}.tif"),
                     np.transpose(rotated, (2, 1, 0)).astype(volume.dtype, copy=False))
    return res


def run_pca_timing(original_path: Path, results_root: Path, annotations: dict, animals: list,
                   n_workers: int, keep_work: bool, eval_range: tuple, all_vols: list) -> dict:
    cfg_dir = results_root / PCA_ID
    work = cfg_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text(json.dumps(
        {"id": PCA_ID, "method": "segment + PCA + Rodrigues (axis alignment only)",
         "qa_render": "removed (timing only)", "views": 6, "grid": P.grid_w,
         "eval_range": list(eval_range), "animals": animals}, indent=2), encoding="utf-8")

    lo, hi = eval_range

    def _prefix(stem):
        return stem.rsplit("_", 1)[0]

    anno_prefixes = {a: {_prefix(Path(fn).stem) for fn in annotations.get(a, {})} for a in animals}
    eval_volumes = sorted({f for f in all_vols
                           if (idx := RE.individual_index(f.stem)) is not None and lo <= idx <= hi
                           and _prefix(f.stem) in anno_prefixes.get(f.parts[-2], set())})
    path_of = {(f.parts[-2], f.stem): f for f in eval_volumes}
    if not eval_volumes:
        print(f"   [{PCA_ID}] WARNING: no individuals with index in {lo}-{hi} for {animals}")
    timings = {}

    # Pass 1: segment (parallel)
    print(f"   Pass 1/3: segment {len(eval_volumes)} volumes on {n_workers} workers", flush=True)
    t0 = time.perf_counter()
    seg_results = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_SPAWN) as ex:
        futs = [ex.submit(_pca_segment_worker, f) for f in eval_volumes]
        for fut in as_completed(futs):
            seg_results.append(fut.result())
    timings["pass1_s"] = time.perf_counter() - t0
    print(f"   Pass 1 done in {timings['pass1_s']:.0f}s", flush=True)

    # Pass 2: PCA principal axis (serial)
    print(f"   Pass 2/3: PCA principal axis for "
          f"{sum(r['status'] == 'ok' for r in seg_results)} volumes", flush=True)
    t0 = time.perf_counter()
    for r in seg_results:
        if r["status"] != "ok":
            continue
        axis, centroid, eval0, elong = _pca_principal_axis(r["coords"])
        if eval0 <= 1e-9:
            r["status"] = "skip_degenerate"
        else:
            r["principal_axis"] = axis.tolist()
            r["centroid"] = centroid.tolist()
            r["elongation"] = elong
            r["target_axis"] = ["+X", "+Y", "+Z"][int(np.argmax(r["shape"]))]
        r["coords"] = None
    timings["pass2_s"] = time.perf_counter() - t0
    print(f"   Pass 2 done in {timings['pass2_s']:.2f}s", flush=True)

    produced = [r for r in seg_results if r["status"] == "ok"]

    # Pass 3: rotate + write (parallel, NO QA render)
    print(f"   Pass 3/3: rotate {len(produced)} volumes on {n_workers} workers (no QA render)", flush=True)
    t0 = time.perf_counter()
    rotate_times = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_SPAWN) as ex:
        futs = [ex.submit(_pca_rotate_worker, path_of[(r["animal"], r["individual"])], work,
                          r["principal_axis"], r["centroid"], r["target_axis"]) for r in produced]
        for fut in as_completed(futs):
            rr = fut.result()
            rotate_times[(rr["animal"], rr["individual"])] = rr["rotate_s"]
    timings["pass3_s"] = time.perf_counter() - t0
    timings["load_s"] = 0.0
    print(f"   Pass 3 done in {timings['pass3_s']:.0f}s", flush=True)

    n_ok = len(produced)
    metrics = {
        "id": PCA_ID, "timings": timings,
        "n_individuals": len(seg_results), "n_ok": n_ok,
        "note": "TIMING ONLY, QA re-render removed.",
        "per_individual": [{"animal": r["animal"], "individual": r["individual"], "status": r["status"],
                            "n_voxels": r["n_voxels"], "elongation": round(r.get("elongation", 0.0), 3),
                            "seg_s": round(r["seg_s"], 3),
                            "rotate_s": round(rotate_times.get((r["animal"], r["individual"]), 0.0), 3)}
                           for r in seg_results],
    }
    (cfg_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return metrics


# =========================================================================== #
# Main
# =========================================================================== #
def _dino_row(params):
    row = {k: "" for k in RE.SUMMARY_FIELDS}
    row.update({k: params.get(k, "") for k in
                ["id", "axis", "level", "views", "model", "grid", "threshold", "top_k", "anno_mode"]})
    return row


def _pca_row():
    row = {k: "" for k in RE.SUMMARY_FIELDS}
    row.update({"id": PCA_ID, "axis": "method", "level": PCA_ID, "views": "6",
                "model": "none", "grid": P.grid_w, "threshold": "otsu"})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default="data/original_photos")
    ap.add_argument("--results", default=str(REPO_ROOT / "HPC_round_5" / "experiments"))
    ap.add_argument("--animals", nargs="+", default=["GH"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--mode", choices=["ofat", "grid"], default="ofat", help="DINO config set")
    ap.add_argument("--eval-range", nargs=2, type=int, default=[11, 20], metavar=("LO", "HI"))
    ap.add_argument("--only", nargs="+", default=None, help="run only these DINO config ids")
    ap.add_argument("--methods", nargs="+", choices=["dino", "pca"], default=["dino", "pca"])
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    original_path = Path(args.original)
    results_root = Path(args.results)
    results_root.mkdir(parents=True, exist_ok=True)

    if args.workers > 0:
        n_workers = args.workers
    else:
        env_cpus = os.environ.get("LSB_DJOB_NUMPROC") or os.environ.get("SLURM_CPUS_PER_TASK")
        n_workers = int(env_cpus) if env_cpus else max(1, (os.cpu_count() or 4) - 2)

    annotations = json.loads(
        (REPO_ROOT / "Annoteringer" / "annotations_output" / "image_annotations.json").read_text(encoding="utf-8"))

    print("Discovering input volumes ...", flush=True)
    all_vols = [f for f in find_input_volumes(original_path) if f.parts[-2] in args.animals]
    print(f"Found {len(all_vols)} volumes for {args.animals} | workers={n_workers} | methods={args.methods}")
    print(f"Results -> {results_root}  (TIMING ONLY, no QA render)\n")

    summary_path = results_root / "summary.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RE.SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()

        # ---- DINO sweep (QA render removed, Pass 1 render kept) --------------
        if "dino" in args.methods:
            configs = RE.build_grid_configs() if args.mode == "grid" else RE.build_ofat_configs()
            if args.only:
                wanted = set(args.only)
                configs = [c for c in configs if c["id"] in wanted]
            print(f"DINO: {len(configs)} configs\n")
            for i, spec in enumerate(configs, 1):
                params, cfg = RE._materialize(spec)
                cfg_id = params["id"]
                print(f"[dino {i}/{len(configs)}] {cfg_id} ...", flush=True)
                row = _dino_row(params)
                t_cfg = time.perf_counter()
                try:
                    m = run_dino_config_timing(params, cfg, original_path, results_root, annotations,
                                               args.animals, n_workers, args.keep_work,
                                               tuple(args.eval_range), all_vols)
                    row.update({"n_individuals": m["n_individuals"], "n_ok": m["n_ok"],
                                "pass1_s": round(m["timings"]["pass1_s"], 2),
                                "pass2_s": round(m["timings"]["pass2_s"], 2),
                                "pass3_s": round(m["timings"]["pass3_s"], 2),
                                "load_s": round(m["timings"]["load_s"], 2),
                                "total_s": round(time.perf_counter() - t_cfg, 2), "status": "ok"})
                    print(f"      done: {m['n_ok']}/{m['n_individuals']} | "
                          f"p1={row['pass1_s']}s p2={row['pass2_s']}s p3={row['pass3_s']}s")
                except Exception as e:
                    row["status"] = f"FAILED: {e}"
                    row["total_s"] = round(time.perf_counter() - t_cfg, 2)
                    print(f"      FAILED: {e}")
                    traceback.print_exc()
                writer.writerow(row)
                fh.flush()

        # ---- PCA (QA render removed) ----------------------------------------
        if "pca" in args.methods:
            print(f"\n[pca] {PCA_ID} ...", flush=True)
            row = _pca_row()
            t_cfg = time.perf_counter()
            try:
                m = run_pca_timing(original_path, results_root, annotations, args.animals,
                                   n_workers, args.keep_work, tuple(args.eval_range), all_vols)
                row.update({"n_individuals": m["n_individuals"], "n_ok": m["n_ok"],
                            "pass1_s": round(m["timings"]["pass1_s"], 2),
                            "pass2_s": round(m["timings"]["pass2_s"], 2),
                            "pass3_s": round(m["timings"]["pass3_s"], 2),
                            "load_s": round(m["timings"]["load_s"], 2),
                            "total_s": round(time.perf_counter() - t_cfg, 2), "status": "ok"})
                print(f"      done: {m['n_ok']}/{m['n_individuals']} | "
                      f"p1={row['pass1_s']}s p2={row['pass2_s']}s p3={row['pass3_s']}s")
            except Exception as e:
                row["status"] = f"FAILED: {e}"
                row["total_s"] = round(time.perf_counter() - t_cfg, 2)
                print(f"      FAILED: {e}")
                traceback.print_exc()
            writer.writerow(row)
            fh.flush()

    print(f"\nSummary (timings, no QA render) written to {summary_path}")


if __name__ == "__main__":
    main()
