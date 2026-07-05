"""
HPC round 4 -- PCA-based reorientation (segment -> PCA -> Rodrigues).

This round replaces the DINOv3 head-localization entirely with a purely geometric
baseline and measures its TIMING and (manual) CORRECTNESS in the same style as the
DINO sweep in ../run_experiments.py -- including the SAME summary.csv columns, so
the two summary files can be stacked / diffed directly.

  1. Segment the specimen with the SAME segmentation the DINO baseline uses
     (pipeline.segment_largest_component -> otsu threshold + dilation + hole-fill +
     intensity-weighted largest connected component).
  2. Run PCA on the segmented voxel coordinates. The LARGEST principal component
     (the eigenvector of the coordinate covariance with the biggest eigenvalue) is
     the long axis of the specimen's body.
  3. Rotate that principal axis onto the volume's largest axis with the Rodrigues
     rotation already in the pipeline (pipeline.rotation_matrix_from_vectors +
     pipeline.rotate_volume), rotating about the segmented centroid.
  4. QA-re-render the rotated volume in the 6 canonical views and stitch a 6-panel
     composite, exactly like the DINO rounds, so correctness is eyeballed the same
     way and configs line up visually.

-----------------------------------------------------------------------------
IMPORTANT: axis alignment only -- no head/tail direction
-----------------------------------------------------------------------------
A PCA principal component is an undirected LINE (defined only up to sign); it does
not know which end is the head. The evaluation individuals are NOT annotated, so
there is no ground-truth head to pick the sign from. This round therefore only
aligns the long axis to the target axis and does NOT resolve head-vs-tail. Expect
roughly half of the specimens to come out pointing head-DOWN -- that is inherent to
PCA without a direction cue, not a bug. `n_ok` below counts specimens that PRODUCED
a rotated output (non-degenerate mask + valid principal axis); whether the head
actually ends up UP is checked by hand in the composites, the same manual count as
the DINO rounds. The eigenvector sign is fixed deterministically (largest-magnitude
component made positive) only for reproducibility; it does not affect the aligned
axis or the manual head-up count.

-----------------------------------------------------------------------------
Timing -- SAME summary.csv columns and 3-pass layout as run_experiments.py
-----------------------------------------------------------------------------
The summary.csv header is imported verbatim from run_experiments.py, so a PCA row
drops straight next to the DINO rows. The three passes are wall-clock timed with
perf_counter and map onto the DINO passes as follows:

  pass1_s : Pass 1 (parallel pool) -- load + segment. The DINO Pass 1 is
            segment + render-6-views; PCA needs NO pre-render, so this pass is
            segment-only and legitimately smaller than the DINO Pass 1.
  pass2_s : Pass 2 (serial, like the DINO GPU pass) -- PCA localization: build the
            coordinate covariance + eigen-decompose + pick the largest principal
            component. This is the geometric stand-in for the DINO Pass 2 (patch
            tokens + prototype + top-k). Pure compute (coords already in memory),
            so it reflects PCA cost, not I/O.
  pass3_s : Pass 3 (parallel pool) -- Rodrigues rotate + write rotated .tif + QA
            re-render the 6 views. Mirrors the DINO Pass 3 (triangulate + rotate +
            QA render); PCA has no triangulation, so only rotate + render remain.
  load_s  : 0 -- there is no model to load (the DINO rounds record the model load
            time here).
  total_s : wall clock for the config. Headline comparison vs. the DINO summary.csv
            (PCA should be far cheaper: no GPU, no model, no tokens, no triangulate,
            and no 6-view pre-render in Pass 1).

The unused DINO knob columns (model / top_k / anno_mode) are blank for a PCA row;
views=6, grid=60 and threshold=otsu are the fixed settings this round uses.

-----------------------------------------------------------------------------
Evaluation set (identical individuals to the earlier rounds)
-----------------------------------------------------------------------------
To compare apples-to-apples, the evaluated individuals are selected EXACTLY as in
run_experiments.py: an individual is evaluated iff (a) it shares a name prefix with
an annotated individual (so it belongs to a prototype id family, e.g. gras_1_*) AND
(b) its trailing index _NNN is in --eval-range (default 11-20). The annotations are
used ONLY for this selection -- PCA itself needs no annotation. The annotated
individuals themselves (e.g. gras_1_000..004) are NOT processed here (there is no
prototype to build), unlike the DINO rounds which render+tokenize them.

-----------------------------------------------------------------------------
Run (from the repo root, same env/venv as the other rounds)
-----------------------------------------------------------------------------
  python HPC_round_4/run_pca_test.py --animals GH \
      --original data/original_photos \
      --results  HPC_round_4/experiments

On the DTU HPC cluster, submit via HPC_round_4/run_pca_test_lsf.sh (bsub); it wraps
this in xvfb-run (needed for the offscreen VTK QA render) and points --original at
the shared BugNIST volumes. No GPU is required for this round, but xvfb IS (the QA
render is the only reason a display is needed).
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

# Make the repo-root modules importable no matter the current working directory:
# this script sits in <repo>/HPC_round_4/, so the repo root is its parent.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline as P  # noqa: E402
from pipeline import (  # noqa: E402
    find_input_volumes, process_volume,
    rotation_matrix_from_vectors, rotate_volume,
)
# Reuse the exact composite montage + individual-index helpers, the 6 canonical
# views, and the summary.csv column set from the DINO harness, so the output lines
# up with (and stacks onto) the earlier rounds.
from run_experiments import (  # noqa: E402
    make_composite, individual_index, VIEWS_6,
    SUMMARY_FIELDS as DINO_FIELDS,
)

# Force 'spawn' for the pools, matching run_experiments.py: spawn re-imports
# pipeline.py with its DEFAULT globals (6 views, grid 60, otsu), which is exactly
# the config this round wants -- no per-worker initializer needed.
_SPAWN = multiprocessing.get_context("spawn")

CONFIG_ID = "pca_seg"


# --------------------------------------------------------------------------- #
# Pool worker functions (top-level so 'spawn' can import them by reference)
# --------------------------------------------------------------------------- #
def _segment_worker(in_path: Path) -> dict:
    """Pass 1 unit: load + segment. Returns the foreground voxel coordinates so
    Pass 2 (PCA) can run without re-reading anything.

    Coordinate convention matches the rest of the project: vedo loads the volume as
    [x, y, z]; segment_largest_component and np.argwhere are in that same voxel xyz
    frame, so the principal axis computed downstream can be rotated onto the target
    axis directly.
    """
    animal, individual = in_path.parts[-2], in_path.stem
    res = {"animal": animal, "individual": individual, "status": "ok",
           "n_voxels": 0, "shape": None, "coords": None, "seg_s": 0.0}

    t = time.perf_counter()
    volume = vedo.load(str(in_path)).tonumpy()          # [x, y, z]
    mask = P.segment_largest_component(volume)          # otsu, largest component
    res["seg_s"] = time.perf_counter() - t

    n = int(mask.sum())
    res["n_voxels"] = n
    res["shape"] = [int(s) for s in volume.shape]
    if n < 10:
        res["status"] = "skip_empty_mask"
        return res
    # int32 is plenty for 512-ish indices and halves the pickled payload sent back
    # to the parent versus argwhere's default int64.
    res["coords"] = np.argwhere(mask).astype(np.int32)   # (N, 3) in [x, y, z]
    return res


def _rotate_render_worker(in_path: Path, work: Path, principal_axis: list,
                          centroid: list, target_axis: str) -> dict:
    """Pass 3 unit: Rodrigues rotate the ORIGINAL volume onto the target axis
    (about the segmented centroid), write the rotated .tif, then QA re-render it in
    the 6 canonical views (pipeline.process_volume, which uses pipeline's default
    6-view / grid-60 config that spawn re-imports fresh in every worker)."""
    animal, individual = in_path.parts[-2], in_path.stem
    res = {"animal": animal, "individual": individual, "rotate_s": 0.0}

    volume = vedo.load(str(in_path)).tonumpy()          # [x, y, z]
    t = time.perf_counter()
    rot = rotation_matrix_from_vectors(np.asarray(principal_axis, dtype=float),
                                       P.AXIS_TO_VECTOR[target_axis])
    rotated = rotate_volume(volume, rot=rot, order=1, center=np.asarray(centroid, dtype=float))
    res["rotate_s"] = time.perf_counter() - t

    out_dir = work / "final" / animal
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{individual}.tif"
    # Store back in the on-disk [z, y, x] page order so the QA re-render's
    # vedo.load(...).tonumpy() reads it back as [x, y, z]. Mirrors the DINO worker.
    tifffile.imwrite(str(out_tif), np.transpose(rotated, (2, 1, 0)).astype(volume.dtype, copy=False))

    process_volume(out_tif, work / "final", work / "final_segmented")  # QA 6-view render
    return res


# --------------------------------------------------------------------------- #
# Pass 2: PCA principal axis (serial, in the parent -- coords already in memory)
# --------------------------------------------------------------------------- #
def _pca_principal_axis(coords: np.ndarray):
    """Largest principal component of the segmented voxel cloud.

    Returns (principal_axis unit vector, centroid, elongation = eval0 / eval1).
    Elongation near 1 means no dominant long axis (a blob) -> ill-conditioned
    direction; reported for the manual read, not used to reject.
    """
    coords = coords.astype(np.float64)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = (centered.T @ centered) / coords.shape[0]      # 3x3 coordinate covariance
    evals, evecs = np.linalg.eigh(cov)                   # ascending eigenvalues
    order = np.argsort(evals)[::-1]                      # largest first
    evals, evecs = evals[order], evecs[:, order]
    principal_axis = evecs[:, 0]
    # Deterministic sign so re-runs are identical. Sign does NOT affect the aligned
    # axis (a line) nor the manual head-up count -- head/tail is not resolved here.
    if principal_axis[int(np.argmax(np.abs(principal_axis)))] < 0.0:
        principal_axis = -principal_axis
    elongation = float(evals[0] / max(evals[1], 1e-12))
    return principal_axis, centroid, evals[0], elongation


# --------------------------------------------------------------------------- #
# One config (pca_seg) end-to-end
# --------------------------------------------------------------------------- #
def run_pca_config(original_path: Path, results_root: Path, annotations: dict,
                   animals: list[str], n_workers: int, keep_work: bool,
                   eval_range: tuple, all_vols: list) -> dict:
    cfg_dir = results_root / CONFIG_ID
    work = cfg_dir / "work"
    comp_dir = cfg_dir / "composites"
    work.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "id": CONFIG_ID,
        "method": "segment + PCA principal axis + Rodrigues (axis alignment only)",
        "segmentation": "otsu / largest connected component (pipeline default)",
        "head_tail_direction": "NOT resolved -- expect ~50% head-down; manual count",
        "views": 6, "grid": P.grid_w,
        "eval_range": list(eval_range), "animals": animals,
    }
    (cfg_dir / "config.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    # Evaluation set: identical selection to run_experiments.py (prefix family +
    # trailing index in eval_range). Annotations are used ONLY to define the family.
    lo, hi = eval_range

    def _prefix(stem: str) -> str:
        return stem.rsplit("_", 1)[0]  # gras_1_015 -> "gras_1", mel_10_002 -> "mel_10"

    anno_prefixes = {a: {_prefix(Path(fn).stem) for fn in annotations.get(a, {})} for a in animals}
    eval_volumes = []
    for f in all_vols:
        idx = individual_index(f.stem)
        if idx is None or not (lo <= idx <= hi):
            continue
        if _prefix(f.stem) in anno_prefixes.get(f.parts[-2], set()):
            eval_volumes.append(f)
    eval_volumes = sorted(set(eval_volumes))
    path_of = {(f.parts[-2], f.stem): f for f in eval_volumes}

    if not eval_volumes:
        print(f"   [{CONFIG_ID}] WARNING: no individuals with index in {lo}-{hi} "
              f"sharing an annotated prefix for {animals}; nothing to do")

    timings = {}

    # --- Pass 1: segment (parallel) ------------------------------------------
    print(f"   Pass 1/3: segment {len(eval_volumes)} volumes on {n_workers} workers", flush=True)
    t0 = time.perf_counter()
    seg_results = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_SPAWN) as ex:
        futs = [ex.submit(_segment_worker, f) for f in eval_volumes]
        for fut in as_completed(futs):
            seg_results.append(fut.result())
    timings["pass1_s"] = time.perf_counter() - t0
    print(f"   Pass 1 done in {timings['pass1_s']:.0f}s", flush=True)

    # --- Pass 2: PCA principal axis (serial, like the DINO GPU pass) ---------
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
        r["coords"] = None  # free the voxel cloud now that the axis is extracted
    timings["pass2_s"] = time.perf_counter() - t0
    print(f"   Pass 2 done in {timings['pass2_s']:.2f}s", flush=True)

    produced = [r for r in seg_results if r["status"] == "ok"]

    # --- Pass 3: rotate + QA render (parallel) -------------------------------
    print(f"   Pass 3/3: rotate + QA render {len(produced)} volumes on {n_workers} workers",
          flush=True)
    t0 = time.perf_counter()
    rotate_times = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_SPAWN) as ex:
        futs = {ex.submit(_rotate_render_worker,
                          path_of[(r["animal"], r["individual"])], work,
                          r["principal_axis"], r["centroid"], r["target_axis"]): r
                for r in produced}
        for fut in as_completed(futs):
            rr = fut.result()
            rotate_times[(rr["animal"], rr["individual"])] = rr["rotate_s"]
    timings["pass3_s"] = time.perf_counter() - t0
    timings["load_s"] = 0.0
    print(f"   Pass 3 done in {timings['pass3_s']:.0f}s", flush=True)

    # --- Build composites (manual correctness check) -------------------------
    qa_labels = [v[0] for v in VIEWS_6]  # always the 6 canonical views
    n_ok = 0
    per_individual = []
    for r in seg_results:
        animal, individual = r["animal"], r["individual"]
        if r["status"] == "ok":
            n_ok += 1
        qa_dir = work / "final_segmented" / animal / individual
        png_paths = [qa_dir / f"{individual}_{lab}.png" for lab in qa_labels]
        header = f"{CONFIG_ID} | {animal}/{individual}"
        if r["status"] != "ok":
            header += f" | {r['status']}"
        else:
            header += f" | axis={r['target_axis']} elong={r['elongation']:.1f}"
        make_composite(png_paths, qa_labels, comp_dir / f"{animal}__{individual}.png", header=header)
        per_individual.append({
            "animal": animal, "individual": individual, "status": r["status"],
            "n_voxels": r["n_voxels"],
            "elongation": round(r.get("elongation", 0.0), 3),
            "target_axis": r.get("target_axis", ""),
            "seg_s": round(r["seg_s"], 3),
            "rotate_s": round(rotate_times.get((animal, individual), 0.0), 3),
        })

    metrics = {
        "id": CONFIG_ID,
        "params": params,
        "timings": timings,
        "n_individuals": len(seg_results),
        "n_ok": n_ok,
        "note": "n_ok = produced a rotated output; head-UP correctness is manual "
                "(see composites). ~50% may be head-down (no head/tail direction). "
                "pass1_s=segment, pass2_s=PCA, pass3_s=rotate+QA-render, load_s=0.",
        "per_individual": per_individual,
    }
    (cfg_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)

    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default="data/original_photos", help="root with <SPECIES>/<ind>.tif")
    ap.add_argument("--results", default=str(REPO_ROOT / "HPC_round_4" / "experiments"),
                    help="output root")
    ap.add_argument("--animals", nargs="+", default=["GH"], help="species codes to evaluate")
    ap.add_argument("--workers", type=int, default=0, help="0 => LSF/SLURM cores or cpu_count-2")
    ap.add_argument("--eval-range", nargs=2, type=int, default=[11, 20], metavar=("LO", "HI"),
                    help="individual index range (trailing _NNN) to evaluate/composite")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep heavy intermediates (rotated .tif + QA renders)")
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
        (REPO_ROOT / "Annoteringer" / "annotations_output" / "image_annotations.json").read_text(encoding="utf-8")
    )

    print("Discovering input volumes ...", flush=True)
    all_vols = [f for f in find_input_volumes(original_path) if f.parts[-2] in args.animals]
    print(f"Found {len(all_vols)} volumes for {args.animals}", flush=True)

    print(f"PCA reorientation test (round 4) | animals={args.animals} | workers={n_workers}")
    print(f"Results -> {results_root}\n")

    # Same summary.csv columns as run_experiments.py (imported verbatim), so the PCA
    # row stacks onto the DINO rows. The knob columns that do not apply to PCA are
    # left blank; the fixed PCA settings (views=6, grid=60, otsu) are filled in.
    summary_path = results_root / "summary.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DINO_FIELDS)
        if write_header:
            writer.writeheader()

        print(f"[1/1] {CONFIG_ID} ...", flush=True)
        row = {k: "" for k in DINO_FIELDS}
        row.update({
            "id": CONFIG_ID, "axis": "method", "level": CONFIG_ID,
            "views": "6", "model": "none", "grid": P.grid_w,
            "threshold": "otsu", "top_k": "", "anno_mode": "",
        })
        t_cfg = time.perf_counter()
        try:
            m = run_pca_config(original_path, results_root, annotations, args.animals,
                               n_workers, args.keep_work, tuple(args.eval_range), all_vols)
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
                  f"check composites under {results_root}/{CONFIG_ID}/composites/")
        except Exception as e:
            row["status"] = f"FAILED: {e}"
            row["total_s"] = round(time.perf_counter() - t_cfg, 2)
            print(f"      FAILED: {e}")
            traceback.print_exc()
        writer.writerow(row)
        fh.flush()

    print(f"\nSummary written to {summary_path}")
    print("Composites for manual correctness counting are under "
          f"{results_root}/{CONFIG_ID}/composites/")
    print("Reminder: head/tail direction is NOT resolved -- count head-UP by hand; "
          "~50% head-down is expected for a pure PCA axis alignment.")


if __name__ == "__main__":
    main()
