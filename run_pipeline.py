from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


MODELS = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
}


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def run(step: str, cmd: list[str], timings: list[tuple[str, float, str]]) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"STEP: {step}", flush=True)
    print(f"{'=' * 60}", flush=True)
    t0 = time.monotonic()
    result = subprocess.run([sys.executable, *cmd])
    elapsed = time.monotonic() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    timings.append((step, elapsed, status))
    print(f"\n[{step}] {status} — {fmt_duration(elapsed)}", flush=True)
    if result.returncode != 0:
        print(f"ERROR: step '{step}' failed. Aborting.", flush=True)
        sys.exit(result.returncode)


def save_report(root: Path, header: str, timings: list[tuple[str, float, str]], total_elapsed: float) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"{header}: {timestamp}",
        f"{'=' * 60}",
    ]
    for step, elapsed, status in timings:
        lines.append(f"  {step:<40} {fmt_duration(elapsed):>10}   {status}")
    lines += [
        f"{'=' * 60}",
        f"  {'TOTAL':<40} {fmt_duration(total_elapsed):>10}",
        "",
    ]
    report = "\n".join(lines)
    print(f"\n{report}", flush=True)

    report_path = root / "pipeline_timings.txt"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_path.write_text(report + existing, encoding="utf-8")
    print(f"Timings saved to {report_path}", flush=True)


def run_full_pipeline(root: Path) -> None:
    timings: list[tuple[str, float, str]] = []
    pipeline_start = time.monotonic()

    # 1. Segment all original volumes and render 6 canonical views per individual.
    run("segment_original_photos", [
        str(root / "segment_original_photos.py"),
        "--input-root", "data/original_photos",
        "--output-root", "data/new_photos/segmented",
        "--overwrite",
        "--max-files", "10",
    ], timings)

    # 2. Render annotated head-marker views for the labelled reference set.
    run("visualize_annotated_heads", [
        str(root / "visualize_annotated_heads.py"),
        "--input-root", "data/original_photos",
        "--output-root", "data/new_photos/head_visualizations",
        "--segmented-root", "data/new_photos/segmented",
        "--max-files", "10"
    ], timings)

    # 3. Find top-3 likely head patches per view using DINOv2 cosine similarity.
    run("top3_head_patches", [
        str(root / "top3_head_patches.py"),
        "--head-vis-root", "data/new_photos/head_visualizations",
        "--segmented-root", "data/new_photos/segmented",
        "--output-root", "data/new_photos/head_top3",
        "--max-files", "10",
        "--model-name", "facebook/dinov2-small",
    ], timings)

    # 4. Fuse head position across 6 views via camera-ray triangulation.
    run("fuse_head_position", [
        str(root / "fuse_head_position.py"),
        "--top3-root", "data/new_photos/head_top3",
        "--segmented-root", "data/new_photos/segmented",
        "--output-root", "data/new_photos/head_fused",
    ], timings)

    # 5. Rotate each volume so the head direction points to +Y.
    run("rotate_head_up", [
        str(root / "rotate_head_up.py"),
        "--fused-root", "data/new_photos/head_fused",
        "--input-root", "data/original_photos",
        "--output-root", "data/finished_photos/rotated",
        "--overwrite",
        "--max-files", "10"
    ], timings)

    # 6. Render 6-view composites of the rotated volumes.
    run("segment_sixview_composite", [
        str(root / "segment_sixview_composite.py"),
        "--input-root", "data/finished_photos/rotated",
        "--output-root", "data/finished_photos/composite",
        "--overwrite",
        "--max-files", "10"
    ], timings)

    save_report(root, "Pipeline run", timings, time.monotonic() - pipeline_start)


def run_benchmark(root: Path, n_animals: int) -> None:
    print(f"\nBenchmarking dinov2-small vs dinov2-base on {n_animals} individuals.", flush=True)
    print("Assumes segmented/ and head_visualizations/ already exist.", flush=True)
    print("Results are written to temporary dirs and do not affect pipeline outputs.\n", flush=True)

    timings: list[tuple[str, float, str]] = []
    benchmark_start = time.monotonic()

    for label, model_id in MODELS.items():
        out_dir = f"data/new_photos/benchmark_top3_{label}"
        run(f"top3_head_patches [{label}]", [
            str(root / "top3_head_patches.py"),
            "--head-vis-root", "data/new_photos/head_visualizations",
            "--segmented-root", "data/new_photos/segmented",
            "--output-root", out_dir,
            "--model-name", model_id,
            "--max-files", str(n_animals),
        ], timings)

    total_elapsed = time.monotonic() - benchmark_start

    # Print comparison summary.
    if len(timings) == 2:
        (_, t_small, _), (_, t_base, _) = timings
        speedup = t_base / t_small if t_small > 0 else float("inf")
        print(f"\n  dinov2-small : {fmt_duration(t_small)}", flush=True)
        print(f"  dinov2-base  : {fmt_duration(t_base)}", flush=True)
        print(f"  speedup      : {speedup:.2f}x (small is faster)", flush=True)

    save_report(root, f"Benchmark run ({n_animals} individuals)", timings, total_elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full insect CT reorientation pipeline, or benchmark DINOv2 model sizes."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Run the full pipeline (default if no command given).")

    bench = subparsers.add_parser("benchmark", help="Compare dinov2-small vs dinov2-base on N individuals.")
    bench.add_argument(
        "--n-animals",
        type=int,
        default=10,
        help="Number of individuals to process per model (default: 10).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).parent

    if args.command == "benchmark":
        run_benchmark(root, args.n_animals)
    else:
        run_full_pipeline(root)


if __name__ == "__main__":
    main()
