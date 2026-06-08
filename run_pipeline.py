from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


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


def main() -> None:
    root = Path(__file__).parent
    timings: list[tuple[str, float, str]] = []
    pipeline_start = time.monotonic()

    # 1. Segment all original volumes and render 6 canonical views per individual.
    run("segment_original_photos", [
        str(root / "segment_original_photos.py"),
        "--input-root", "data/original_photos",
        "--output-root", "data/new_photos/segmented",
        "--overwrite",
    ], timings)

    # 2. Render annotated head-marker views for the labelled reference set.
    run("visualize_annotated_heads", [
        str(root / "visualize_annotated_heads.py"),
        "--input-root", "data/original_photos",
        "--output-root", "data/new_photos/head_visualizations",
        "--segmented-root", "data/new_photos/segmented",
    ], timings)

    # 3. Find top-3 likely head patches per view using DINOv2 cosine similarity.
    run("top3_head_patches", [
        str(root / "top3_head_patches.py"),
        "--head-vis-root", "data/new_photos/head_visualizations",
        "--segmented-root", "data/new_photos/segmented",
        "--output-root", "data/new_photos/head_top3",
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
    ], timings)

    # 6. Render 6-view composites of the rotated volumes.
    run("segment_sixview_composite", [
        str(root / "segment_sixview_composite.py"),
        "--input-root", "data/finished_photos/rotated",
        "--output-root", "data/finished_photos/composite",
        "--overwrite",
    ], timings)

    total_elapsed = time.monotonic() - pipeline_start
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"Pipeline run: {timestamp}",
        f"{'=' * 60}",
    ]
    for step, elapsed, status in timings:
        lines.append(f"  {step:<35} {fmt_duration(elapsed):>10}   {status}")
    lines += [
        f"{'=' * 60}",
        f"  {'TOTAL':<35} {fmt_duration(total_elapsed):>10}",
        "",
    ]
    report = "\n".join(lines)
    print(f"\n{report}", flush=True)

    report_path = root / "pipeline_timings.txt"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_path.write_text(report + existing, encoding="utf-8")
    print(f"Timings saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
