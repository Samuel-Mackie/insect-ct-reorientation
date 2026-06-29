"""
HPC round 3 -- segmentation-threshold test for percentiles 99 and 99.5.

This is a thin wrapper around the existing harness in ../run_experiments.py.
It does NOT modify run_experiments.py or pipeline.py: it imports them, registers
two extra percentile levels (99 and 99.5) on top of the levels round 1/2 already
swept (otsu, 96, 97, 98), and then runs three configs through the unchanged
run_experiments.main(): "baseline" (= otsu at baseline knobs) plus threshold=99
and threshold=99.5. The baseline run is a within-job TIMING ANCHOR -- measured on
the same node in the same job as 99/99.5 -- so a timing gap versus round 1/2's
baseline exposes node drift rather than a real threshold effect. Everything else
-- the OFAT baseline knobs (6 views, base model, grid 60, top_k 5, 5_animals
prototype), the segment/render/tokenize/triangulate/rotate passes, the per-config
composites and summary.csv -- is reused verbatim, so results line up with the
earlier rounds.

----------------------------------------------------------------------------
How segmentation works, and what the threshold does
----------------------------------------------------------------------------
Segmentation lives in pipeline.segment_largest_component(volume). For the sweep,
run_experiments.apply_config() swaps in an equivalent closure that takes the
threshold from the config. The steps are:

  1. Pick an intensity cutoff t:
       - "otsu"      -> skimage.filters.threshold_otsu(volume)  (automatic)
       - a percentile p (96/97/98/99/99.5) -> np.percentile(volume, p)
     A PERCENTILE threshold means "keep only the brightest (100 - p)% of voxels":
     p=99 keeps the top 1% by intensity, p=99.5 keeps the top 0.5%. Higher
     percentile => stricter cutoff => less foreground kept. (These are percentiles
     of intensity, NOT absolute grey values.)
  2. mask = volume > t                          (binary foreground)
  3. ndimage.binary_dilation(mask, iterations=5)  (close small gaps / reconnect)
  4. ndimage.binary_fill_holes(mask)              (fill interior cavities)
  5. ndimage.label(mask) -> connected components
  6. Keep the LARGEST component, weighted by intensity
     (ndimage.sum_labels(volume, labeled, ...) -> argmax) and return it as the mask.

So YES: thresholding still uses the largest-connected-component step. The only
thing the threshold changes is the initial cutoff in step 1; steps 2-6 (dilation,
hole-fill, and the intensity-weighted largest-component selection) are identical
across all threshold levels. With a very strict cutoff (99 / 99.5) the initial
mask is sparser, which is exactly why steps 3-4 (dilation + fill) and the
largest-component pick matter more -- they reconnect the bright fragments of the
specimen into one body before the largest component is taken.

----------------------------------------------------------------------------
Run (from the repo root, same env/venv as the other rounds)
----------------------------------------------------------------------------
  python HPC_round_3/run_threshold_test.py --animals GH \
      --original data/original_photos \
      --results  HPC_round_3/experiments

On the DTU HPC cluster, submit via HPC_round_3/run_threshold_test_lsf.sh (bsub);
it wraps this in xvfb-run and points --original at the shared BugNIST volumes.

All flags of run_experiments.py are accepted and forwarded EXCEPT --mode and
--only, which this wrapper fixes to the two threshold configs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo-root modules importable no matter the current working directory:
# this script sits in <repo>/HPC_round_3/, so the repo root is its parent.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_experiments as RE  # noqa: E402  (import after sys.path tweak)

# The two new threshold levels for this round. Keys are strings to match the
# existing THRESHOLDS keys ("otsu", "96", ...); values are the percentile passed
# to np.percentile (np.percentile accepts the float 99.5 directly).
NEW_THRESHOLDS = {"99": 99, "99.5": 99.5}

# Config ids that build_ofat_configs() will produce. We also run "baseline"
# (which IS the otsu threshold at all-baseline knobs): it is measured in the SAME
# job/node as 99 and 99.5, so it is a within-job timing ANCHOR. Any timing gap
# between baseline here and baseline/threshold=* in round 1/2 reveals node drift
# rather than a real effect of the threshold.
ONLY_IDS = ["baseline"] + [f"threshold={lvl}" for lvl in NEW_THRESHOLDS]
# -> ["baseline", "threshold=99", "threshold=99.5"]   (baseline == otsu)


def register_thresholds() -> None:
    """Add 99 / 99.5 to the harness's threshold tables (in-memory only).

    - THRESHOLDS  : maps a level key -> percentile value used at segmentation time.
    - SWEEPS["threshold"] : the OFAT levels; appending makes build_ofat_configs()
      emit a `threshold=99` / `threshold=99.5` config, selectable via --only.
    We append rather than replace so the ids stay identical to round 1/2.
    """
    for key, pct in NEW_THRESHOLDS.items():
        RE.THRESHOLDS[key] = pct
        if key not in RE.SWEEPS["threshold"]:
            RE.SWEEPS["threshold"].append(key)


def _strip_flags(argv: list[str], multi_value: set[str]) -> list[str]:
    """Drop the given flags and their value(s) from argv.

    Flags in `multi_value` consume every following token until the next `--flag`
    (e.g. --only takes a list); any other listed flag consumes one value.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in multi_value:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        i += 1
        out.append(tok)
    return out


def main() -> None:
    register_thresholds()

    # Forward the user's CLI to run_experiments.main(), but force the run to our
    # fixed config set (baseline + 99 + 99.5) -- so strip any --mode / --only.
    forwarded = _strip_flags(sys.argv[1:], multi_value={"--mode", "--only"})

    # Default the results dir into HPC_round_3/ if the user didn't pass one.
    if "--results" not in forwarded:
        forwarded += ["--results", str(REPO_ROOT / "HPC_round_3" / "experiments")]

    sys.argv = ["run_experiments.py", "--mode", "ofat", "--only", *ONLY_IDS, *forwarded]
    print(f"[HPC_round_3] threshold test -> configs {ONLY_IDS}")
    print(f"[HPC_round_3] forwarding: {' '.join(sys.argv[1:])}\n")
    RE.main()


if __name__ == "__main__":
    main()
