"""
Parallel runner for the insect CT reorientation pipeline.

Method: the heavy, independent functions live in pipeline.py (importable), so a
ProcessPoolExecutor can pickle them by reference under Windows 'spawn'. The two
CPU-bound, single-threaded stages are fanned out across processes:

  Pass 1  segment + VTK render (+ head projection)   -> parallel across volumes
  Pass 2  DINO tokens + prototype + top-k            -> SERIAL (torch already
                                                         uses all cores; the
                                                         model is loaded once)
  Pass 3  triangulate + rotate (+ QA render)          -> parallel across volumes

The notebook (full_pipeline_dinov3.ipynb) is left untouched; this is a separate
entry point. Run from the repo root:  python run_parallel.py
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from pipeline import (
    VIEWS, AXIS_TO_VECTOR, animals,
    ransac_threshold, ransac_iterations,
    find_input_volumes, process_volume, get_head_information,
    load_dino_model, extract_patch_tokens,
    discover_segmented_individuals, build_prototype_animal, save_top_k_patches,
    build_rays_from_individual, ransac_fuse,
    rotation_matrix_from_vectors, rotate_volume,
)
import numpy as np
import vedo
import tifffile


# --- worker functions (must be top-level so spawn can import them) -----------

def segment_one(in_path, in_root, out_root):
    """Pass 1 unit of work: segment + render one volume (the expensive part)."""
    process_volume(in_path, in_root, out_root)
    return f"[seg] {in_path.parts[-2]}/{in_path.stem}"


def rotate_one(animal, individual, original_path, info_path, do_qa):
    """Pass 3 unit of work: fuse head rays, rotate volume to +Y, optional QA render."""
    output_root = info_path / "segmented"
    tokens_root = info_path / "tokens"

    origins, dirs = build_rays_from_individual(output_root, tokens_root, animal, individual)
    if len(origins) < 2:
        return f"[{animal} {individual}] skip - not enough rays"

    result = ransac_fuse(origins, dirs, threshold=ransac_threshold, refine_iters=ransac_iterations)
    if result is None:
        return f"[{animal} {individual}] skip - RANSAC failed"

    head_xyz = result["point"]
    meta = json.loads((output_root / animal / individual / "metadata.json").read_text(encoding="utf-8"))
    com_xyz = np.array(meta["center_of_mass_xyz"], dtype=float)

    # Persist the fusion result for diagnostics / reproducibility
    (output_root / animal / individual / "head_fused.json").write_text(
        json.dumps({
            "head_xyz": head_xyz.tolist(),
            "com_xyz": com_xyz.tolist(),
            "n_inliers": result["n_inliers"],
            "n_rays": len(origins),
        }, indent=2),
        encoding="utf-8",
    )

    volume = vedo.load(str(original_path / animal / f"{individual}.tif")).tonumpy()
    # Point the head along the largest volume axis so the elongated body fits the
    # array bounds (affine_transform keeps the output shape == input shape).
    target_axis = ["+X", "+Y", "+Z"][int(np.argmax(meta["volume_shape"]))]
    rot = rotation_matrix_from_vectors(head_xyz - com_xyz, AXIS_TO_VECTOR[target_axis])
    rotated = rotate_volume(volume, rot=rot, order=1, center=com_xyz)

    out_tif_dir = info_path / "final" / animal
    out_tif_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_tif_dir / f"{individual}.tif"
    # vedo.load reverses tifffile's axis order, so the rotation runs in vedo's [x,y,z]
    # frame but tifffile.imwrite would store x-as-pages. Transpose to z-as-pages so the
    # round-trip is identity: vedo.load(out_tif) -> [x,y,z] with the head still on +Z.
    tifffile.imwrite(str(out_tif), np.transpose(rotated, (2, 1, 0)).astype(volume.dtype, copy=False))

    if do_qa:
        process_volume(out_tif, info_path / "final", info_path / "final_segmented")
    return f"[done] {animal}/{individual}"


# --- orchestration -----------------------------------------------------------

def main(original_path, info_path, n_workers=6, do_qa=True):
    info_path.mkdir(parents=True, exist_ok=True)
    input_root = original_path
    output_root = info_path / "segmented"
    tokens_root = info_path / "tokens"

    annotations = json.loads(
        Path("Annoteringer/annotations_output/image_annotations.json").read_text(encoding="utf-8")
    )
    volumes = [f for f in find_input_volumes(original_path) if f.parts[-2] in animals]

    # Pass 1 (parallel): segment + render. Comment this block out to reuse an
    # existing segmentation (the PNGs + metadata.json under info_path/segmented).
    # print(f"Pass 1: segment + render | {len(volumes)} volumes | {n_workers} workers")
    # with ProcessPoolExecutor(max_workers=n_workers) as ex:
    #     futs = [ex.submit(segment_one, f, input_root, output_root) for f in volumes]
    #     for fut in as_completed(futs):
    #         print("  ", fut.result())

    # print("Pass 1b: head projection")
    # for f in volumes:
    #     anno = annotations.get(f.parts[-2], {}).get(f.name)
    #     if anno is None:
    #         print(f"   no annotation for {f.parts[-2]}/{f.name} - skipping")
    #         continue
    #     get_head_information(f, input_root, output_root, anno)

    # # Pass 2 (serial): DINO tokens use all cores, so one process. Model loaded once.
    # print("Pass 2: DINO tokens + prototype + top-k")
    # processor, model, device = load_dino_model()
    # for f in volumes:
    #     animal, individual = f.parts[-2], f.stem
    #     for angle, *_ in VIEWS:
    #         img = output_root / animal / individual / f"{individual}_{angle}.png"
    #         extract_patch_tokens(img, output_root, tokens_root, processor, model, device)
    # for animal in animals:
    #     try:
    #         build_prototype_animal(animal, tokens_root, output_root)
    #     except ValueError as e:
    #         print("  ", e)
    #         continue
    #     save_top_k_patches(tokens_root, output_root, animal, k=4)
    #     print(f"   {animal} - finished saving patches")

    # Pass 3 (parallel): triangulate + rotate + optional QA render
    jobs = [(a, ind) for a in animals for ind in discover_segmented_individuals(output_root, a)]
    print(f"Pass 3: rotate + QA | {len(jobs)} individuals | {n_workers} workers")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(rotate_one, a, ind, original_path, info_path, do_qa) for a, ind in jobs]
        for fut in as_completed(futs):
            print("  ", fut.result())

    print("Done.")


if __name__ == "__main__":
    original_path = Path("data/original_photos")
    info_path = Path("data/test_dinov3_parallel")
    # 8 cores -> leave headroom for the OS / disk I/O. Lower this if you hit a
    # memory ceiling (each worker holds a full volume in RAM).
    main(original_path, info_path, n_workers=6, do_qa=True)
