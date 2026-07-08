"""
HPC round 5 -- percentile thresholds 99 and 99.5, render-free TIMING.

Thin wrapper (same idea as HPC_round_3/run_threshold_test.py, but for the
render-free timing harness): it registers the two extra percentile threshold levels
99 and 99.5 on top of the base levels (otsu / 96 / 97 / 98) and then runs ONLY those
two DINO configs through run_timing_noqa.py -- i.e. WITHOUT the QA re-render. The
resulting threshold=99 / threshold=99.5 rows land in the same round-5 summary.csv
(per species) and in the same format as the rest of round 5, so they slot straight
into the timing table.

It edits nothing: pipeline.py, run_experiments.py and run_timing_noqa.py are all
imported and reused verbatim. As in round 3, the threshold VALUE reaches the pool
workers purely as data (cfg["seg_threshold"], applied by run_experiments.apply_config
inside each spawned worker), so the in-parent registration below is all that is
needed -- the workers never read the THRESHOLDS table themselves.

Why only 99 / 99.5: they were introduced in round 3's wrapper and are NOT part of
run_experiments' default OFAT sweep, so the round-5 run (which used the defaults)
did not include them. This wrapper fills exactly that gap, render-free.

Run (repo root, same env/venv as the other rounds)
--------------------------------------------------
  python HPC_round_5/run_timing_noqa_thresholds.py --animals GH \
      --original data/original_photos \
      --results  HPC_round_5/experiments

On the DTU HPC cluster, submit via HPC_round_5/run_timing_noqa_thresholds_lsf.sh
(bsub): it requests the gpuv100 queue + a GPU (needed for DINO Pass 2) and wraps the
run in xvfb-run (DINO's Pass 1 render still runs). All run_timing_noqa.py flags are
forwarded EXCEPT --mode / --only / --methods, which this wrapper fixes to the two
threshold configs (DINO only).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # so `import run_timing_noqa` works from any cwd

import run_experiments as RE      # noqa: E402
import run_timing_noqa as T       # noqa: E402

# New percentile levels for this round. Keys are strings to match the existing
# THRESHOLDS keys; values are the percentile passed to np.percentile.
NEW_THRESHOLDS = {"99": 99, "99.5": 99.5}
ONLY_IDS = [f"threshold={lvl}" for lvl in NEW_THRESHOLDS]  # -> threshold=99, threshold=99.5


def register_thresholds() -> None:
    """Add 99 / 99.5 to run_experiments' threshold tables (in-memory, parent only).

    THRESHOLDS maps level key -> percentile used at segmentation time; appending to
    SWEEPS["threshold"] makes RE.build_ofat_configs() emit the two extra configs
    (selectable via --only). We append (not replace) so all other ids stay identical.
    """
    for key, pct in NEW_THRESHOLDS.items():
        RE.THRESHOLDS[key] = pct
        if key not in RE.SWEEPS["threshold"]:
            RE.SWEEPS["threshold"].append(key)


def _strip_flags(argv: list, multi_value: set) -> list:
    """Drop the given flags and their value(s) from argv (multi_value flags consume
    every following token until the next --flag)."""
    out, i = [], 0
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

    forwarded = _strip_flags(sys.argv[1:], multi_value={"--mode", "--only", "--methods"})
    if "--results" not in forwarded:
        forwarded += ["--results", str(REPO_ROOT / "HPC_round_5" / "experiments")]

    # DINO only (thresholds are a DINO knob), OFAT config set, just the two levels.
    sys.argv = ["run_timing_noqa.py", "--methods", "dino", "--mode", "ofat",
                "--only", *ONLY_IDS, *forwarded]
    print(f"[HPC_round_5/thresholds] render-free timing -> configs {ONLY_IDS}")
    print(f"[HPC_round_5/thresholds] forwarding: {' '.join(sys.argv[1:])}\n")
    T.main()


if __name__ == "__main__":
    main()
