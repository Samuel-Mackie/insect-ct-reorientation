from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(step: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"STEP: {step}", flush=True)
    print(f"{'=' * 60}", flush=True)
    result = subprocess.run([sys.executable, *cmd])
    if result.returncode != 0:
        print(f"\nERROR: step '{step}' failed with exit code {result.returncode}. Aborting.", flush=True)
        sys.exit(result.returncode)


def main() -> None:
    root = Path(__file__).parent

    # 1. Segment all original volumes and render 6 canonical views per individual.
    run("segment_original_photos", [
        str(root / "segment_original_photos.py"),
        "--input-root", "data/original_photos",
        "--output-root", "data/new_photos/segmented",
        "--overwrite",
    ])

    # 2. Render annotated head-marker views for the labelled reference set.
    run("visualize_annotated_heads", [
        str(root / "visualize_annotated_heads.py"),
        "--input-root", "data/original_photos",
        "--output-root", "data/new_photos/head_visualizations",
        "--segmented-root", "data/new_photos/segmented",
    ])

    # 3. Find top-3 likely head patches per view using DINOv2 cosine similarity.
    run("top3_head_patches", [
        str(root / "top3_head_patches.py"),
        "--head-vis-root", "data/new_photos/head_visualizations",
        "--segmented-root", "data/new_photos/segmented",
        "--output-root", "data/new_photos/head_top3",
    ])

    # 4. Fuse head position across 6 views via camera-ray triangulation.
    run("fuse_head_position", [
        str(root / "fuse_head_position.py"),
        "--top3-root", "data/new_photos/head_top3",
        "--segmented-root", "data/new_photos/segmented",
        "--output-root", "data/new_photos/head_fused",
    ])

    # 5. Rotate each volume so the head direction points to +Y.
    run("rotate_head_up", [
        str(root / "rotate_head_up.py"),
        "--fused-root", "data/new_photos/head_fused",
        "--input-root", "data/original_photos",
        "--output-root", "data/finished_photos/rotated",
        "--overwrite",
    ])

    # 6. Render 6-view composites of the rotated volumes.
    run("segment_sixview_composite", [
        str(root / "segment_sixview_composite.py"),
        "--input-root", "data/finished_photos/rotated",
        "--output-root", "data/finished_photos/composite",
        "--overwrite",
    ])

    print("\nPipeline complete.", flush=True)


if __name__ == "__main__":
    main()
