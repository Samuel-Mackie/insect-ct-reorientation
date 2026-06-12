from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel


ANGLE_TO_INDEX = {
    "+X": 1,
    "-X": 2,
    "+Y": 3,
    "-Y": 4,
    "+Z": 5,
    "-Z": 6,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find top-3 likely head patches for each image using cosine similarity to annotated head prototypes."
    )
    parser.add_argument("--animal", type=str, default=None, help="Species code (e.g. AC). Omit for all.")
    parser.add_argument("--angle", type=str, default=None, help="One angle (+X,-X,+Y,-Y,+Z,-Z). Omit for all.")
    parser.add_argument(
        "--head-vis-root",
        type=Path,
        default=Path("data/new_photos/head_visualizations"),
        help="Root containing head_projection.json per annotated individual.",
    )
    parser.add_argument(
        "--segmented-root",
        type=Path,
        default=Path("data/new_photos/segmented"),
        help="Root containing segmented images to score.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/new_photos/head_top3"),
        help="Root output folder.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-small",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--explained-variance",
        type=float,
        default=None,
        help="Optional PCA cumulative explained variance in (0,1). Omit to use all components.",
    )
    parser.add_argument("--bg-threshold", type=int, default=240, help="White-background threshold.")
    parser.add_argument(
        "--fg-patch-ratio",
        type=float,
        default=0.20,
        help="Minimum foreground pixel ratio in a patch.",
    )
    parser.add_argument(
        "--min-patch-distance",
        type=float,
        default=2.0,
        help="Minimum patch-grid distance between ranked top candidates.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of ranked patch candidates to save/visualize.",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/cuda:0.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on total individuals processed (useful for benchmarks).",
    )
    return parser.parse_args()


def discover_animals(head_vis_root: Path) -> list[str]:
    return sorted(p.name for p in head_vis_root.iterdir() if p.is_dir())


def discover_segmented_individuals(segmented_root: Path, animal: str) -> list[str]:
    species_dir = segmented_root / animal
    if not species_dir.exists():
        return []
    return sorted(p.name for p in species_dir.iterdir() if p.is_dir())


def normalize_angle(angle: str) -> str:
    if angle not in ANGLE_TO_INDEX:
        raise ValueError(f"Invalid angle '{angle}'. Valid: {list(ANGLE_TO_INDEX.keys())}")
    return angle


def load_head_annotations(
    head_vis_root: Path,
    animal: str,
    angle: str,
    segmented_root: Path,
) -> list[dict[str, object]]:
    animal_dir = head_vis_root / animal
    if not animal_dir.exists():
        return []

    idx = ANGLE_TO_INDEX[angle]
    entries: list[dict[str, object]] = []
    for meta_path in sorted(animal_dir.rglob("head_projection.json")):
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        view = data.get("views", {}).get(angle)
        if view is None:
            continue
        scan_stem = meta_path.parent.name
        segmented_img = segmented_root / animal / scan_stem / f"{scan_stem}_{angle}.png"
        if not segmented_img.exists():
            continue
        entries.append(
            {
                "scan_stem": scan_stem,
                "segmented_image": segmented_img,
                "patch_row": int(view["patch_row"]),
                "patch_col": int(view["patch_col"]),
            }
        )
    return entries


def crop_to_patch_multiple(image: Image.Image, patch_size: int) -> Image.Image:
    w, h = image.size
    nw = (w // patch_size) * patch_size
    nh = (h // patch_size) * patch_size
    if nw <= 0 or nh <= 0:
        raise ValueError(f"Image too small for patch_size={patch_size}: {(w, h)}")
    if (nw, nh) == (w, h):
        return image
    return image.crop((0, 0, nw, nh))


def make_foreground_patch_mask(
    image: Image.Image,
    patch_size: int,
    bg_threshold: int,
    fg_patch_ratio: float,
) -> tuple[np.ndarray, int, int]:
    img = np.asarray(image)
    h, w = img.shape[:2]
    gh, gw = h // patch_size, w // patch_size
    bg = np.all(img > bg_threshold, axis=2)
    fg_pixel = ~bg
    fg_patch = np.zeros((gh, gw), dtype=bool)
    for r in range(gh):
        for c in range(gw):
            p = fg_pixel[r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size]
            fg_patch[r, c] = p.mean() > fg_patch_ratio
    return fg_patch.reshape(-1), gh, gw


def extract_patch_tokens(
    image_path: Path,
    processor: AutoImageProcessor,
    model: AutoModel,
    patch_size: int,
    bg_threshold: int,
    fg_patch_ratio: float,
    device: str,
) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    image = crop_to_patch_multiple(image, patch_size)
    fg_flat, gh, gw = make_foreground_patch_mask(image, patch_size, bg_threshold, fg_patch_ratio)

    with torch.inference_mode():
        inputs = processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        tokens = outputs.last_hidden_state[:, 1:, :].squeeze(0).detach().cpu().numpy()

    if tokens.shape[0] != gh * gw:
        raise ValueError(f"Patch count mismatch in {image_path}: got {tokens.shape[0]}, expected {gh*gw}")
    return {"image": image, "tokens": tokens, "fg_flat": fg_flat, "grid_h": gh, "grid_w": gw}


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-8)


def choose_components_by_variance(all_tokens: np.ndarray, threshold: float) -> tuple[PCA, int]:
    pca_full = PCA(n_components=min(all_tokens.shape[0], all_tokens.shape[1]))
    pca_full.fit(all_tokens)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n = int(np.searchsorted(cum, threshold, side="left") + 1)
    n = max(1, min(n, all_tokens.shape[1]))
    pca = PCA(n_components=n)
    pca.fit(all_tokens)
    return pca, n


def pick_top_k_peaks(
    sim_grid: np.ndarray,
    fg_grid: np.ndarray,
    k: int,
    min_patch_distance: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[int, int, float]] = []
    for r in range(sim_grid.shape[0]):
        for c in range(sim_grid.shape[1]):
            if fg_grid[r, c]:
                candidates.append((r, c, float(sim_grid[r, c])))
    candidates.sort(key=lambda x: x[2], reverse=True)

    chosen: list[tuple[int, int, float]] = []
    for r, c, s in candidates:
        ok = True
        for rr, cc, _ in chosen:
            if np.hypot(r - rr, c - cc) < min_patch_distance:
                ok = False
                break
        if ok:
            chosen.append((r, c, s))
        if len(chosen) >= k:
            break
    return chosen


def patch_rect(row: int, col: int, patch_size: int) -> tuple[int, int, int, int]:
    x0 = col * patch_size
    y0 = row * patch_size
    x1 = x0 + patch_size - 1
    y1 = y0 + patch_size - 1
    return x0, y0, x1, y1


def load_volume_shape(segmented_root: Path, animal: str, individual: str) -> list[int] | None:
    meta_path = segmented_root / animal / individual / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        shape = data.get("volume_shape_xyz")
        if not (isinstance(shape, list) and len(shape) == 3):
            shape = data.get("volume_shape")
        if isinstance(shape, list) and len(shape) == 3:
            return [int(shape[0]), int(shape[1]), int(shape[2])]
    except Exception:
        return None
    return None


def draw_topk_overlay(
    image: Image.Image,
    peaks: list[tuple[int, int, float]],
    patch_size: int,
) -> Image.Image:
    colors = [(255, 0, 0), (255, 140, 0), (0, 170, 255), (0, 200, 100), (200, 0, 255)]
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, (r, c, s) in enumerate(peaks, start=1):
        color = colors[(i - 1) % len(colors)]
        rect = patch_rect(r, c, patch_size)
        draw.rectangle(rect, outline=color, width=3)
        label = f"{i}:{s:.3f}"
        tx, ty = rect[0] + 2, max(0, rect[1] - 14)
        draw.rectangle((tx - 1, ty - 1, tx + 84, ty + 11), fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=color)
    return out


def main() -> None:
    args = parse_args()
    if args.explained_variance is not None and not (0.0 < args.explained_variance <= 1.0):
        raise ValueError("--explained-variance must be in (0,1].")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Starting top3 head patch run | model={args.model_name} | device={device}")
    log("Loading DINO processor...")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    log("Loading DINO model...")
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    log("Model loaded.")
    patch_size = int(getattr(model.config, "patch_size", 14))

    animals = [args.animal] if args.animal else discover_animals(args.head_vis_root)
    angles = [normalize_angle(args.angle)] if args.angle else list(ANGLE_TO_INDEX.keys())

    token_cache: dict[Path, dict[str, object]] = {}
    total = 0
    processed_individuals = 0

    for animal_idx, animal in enumerate(animals, start=1):
        if args.max_files is not None and processed_individuals >= args.max_files:
            break
        log(f"Animal {animal_idx}/{len(animals)}: {animal}")
        angle_models: dict[str, dict[str, object]] = {}
        for angle in angles:
            log(f"  Preparing prototype for angle {angle}")
            annotated = load_head_annotations(args.head_vis_root, animal, angle, args.segmented_root)
            if not annotated:
                log(f"  Preparing prototype for angle {angle}: no annotations found")
                continue
            by_path = {Path(e["segmented_image"]): e for e in annotated}  # type: ignore[index]

            to_extract = [p for p in by_path if p not in token_cache]
            if to_extract:
                log(f"  Preparing prototype for angle {angle}: extracting {len(to_extract)} annotated token set(s)")
            for p_idx, p in enumerate(to_extract, start=1):
                log(f"    Prototype tokens {p_idx}/{len(to_extract)}: {p.name}")
                if p not in token_cache:
                    token_cache[p] = extract_patch_tokens(
                        p,
                        processor,
                        model,
                        patch_size,
                        args.bg_threshold,
                        args.fg_patch_ratio,
                        device,
                    )

            proto_list: list[np.ndarray] = []
            pca_fit_list: list[np.ndarray] = []
            for p, entry in by_path.items():
                feats = token_cache[p]
                tok = feats["tokens"]  # type: ignore[assignment]
                fg = feats["fg_flat"]  # type: ignore[assignment]
                gw = int(feats["grid_w"])  # type: ignore[arg-type]
                gh = int(feats["grid_h"])  # type: ignore[arg-type]
                r = int(entry["patch_row"])  # type: ignore[arg-type]
                c = int(entry["patch_col"])  # type: ignore[arg-type]
                if 0 <= r < gh and 0 <= c < gw:
                    proto_list.append(tok[r * gw + c])
                    pca_fit_list.append(tok[fg])  # type: ignore[index]

            if not proto_list:
                log(f"  Preparing prototype for angle {angle}: no valid prototype patches")
                continue

            prototype = np.mean(np.stack(proto_list, axis=0), axis=0, keepdims=True)
            pca = None
            mode = "all_components"
            n_components = int(prototype.shape[1])
            if args.explained_variance is not None and args.explained_variance < 1.0:
                log(f"  Preparing prototype for angle {angle}: fitting PCA ({args.explained_variance:.3f})")
                fit_matrix = np.concatenate(pca_fit_list, axis=0)
                pca, n_components = choose_components_by_variance(fit_matrix, args.explained_variance)
                mode = f"pca_{args.explained_variance:.3f}"

            angle_models[angle] = {
                "prototype": prototype,
                "pca": pca,
                "mode": mode,
                "n_components": n_components,
                "prototype_count": len(proto_list),
            }
            log(f"  Preparing prototype for angle {angle}: ready (prototype_count={len(proto_list)})")

        if not angle_models:
            log("  No valid angle prototypes found, skipping animal")
            continue

        individuals = discover_segmented_individuals(args.segmented_root, animal)
        log(f"  Found {len(individuals)} individual(s)")
        for i_idx, individual in enumerate(individuals, start=1):
            if args.max_files is not None and processed_individuals >= args.max_files:
                break
            log(f"  Individual {i_idx}/{len(individuals)}: {individual}")
            for angle in angles:
                if angle not in angle_models:
                    log(f"    Angle {angle}: no prototype")
                    continue

                query = args.segmented_root / animal / individual / f"{individual}_{angle}.png"
                if not query.exists():
                    log(f"    Angle {angle}: missing image")
                    continue

                log(f"    Angle {angle}: running")
                if query not in token_cache:
                    token_cache[query] = extract_patch_tokens(
                        query,
                        processor,
                        model,
                        patch_size,
                        args.bg_threshold,
                        args.fg_patch_ratio,
                        device,
                    )
                q = token_cache[query]
                q_tokens = q["tokens"]  # type: ignore[assignment]
                q_fg = q["fg_flat"]  # type: ignore[assignment]
                q_gh = int(q["grid_h"])  # type: ignore[arg-type]
                q_gw = int(q["grid_w"])  # type: ignore[arg-type]
                q_image: Image.Image = q["image"]  # type: ignore[assignment]

                model_info = angle_models[angle]
                prototype = model_info["prototype"]  # type: ignore[assignment]
                pca = model_info["pca"]  # type: ignore[assignment]

                if pca is not None:
                    proto_m = pca.transform(prototype)
                    q_m = pca.transform(q_tokens)
                else:
                    proto_m = prototype
                    q_m = q_tokens

                proto_m = l2_normalize(proto_m, axis=1)
                q_m = l2_normalize(q_m, axis=1)
                sims = (q_m @ proto_m.T).squeeze(1)
                sim_grid = sims.reshape(q_gh, q_gw)
                fg_grid = q_fg.reshape(q_gh, q_gw)

                peaks = pick_top_k_peaks(
                    sim_grid=sim_grid,
                    fg_grid=fg_grid,
                    k=args.top_k,
                    min_patch_distance=args.min_patch_distance,
                )
                overlay = draw_topk_overlay(q_image, peaks, patch_size=patch_size)

                base_dir = args.output_root / animal / individual
                viz_dir = base_dir / "visualisations"
                data_dir = base_dir / "data"
                json_dir = base_dir / "json"
                viz_dir.mkdir(parents=True, exist_ok=True)
                data_dir.mkdir(parents=True, exist_ok=True)
                json_dir.mkdir(parents=True, exist_ok=True)

                stem = query.stem
                overlay_path = viz_dir / f"{stem}_top{args.top_k}_heads.png"
                map_path = data_dir / f"{stem}_cosine_patch_map.npy"
                json_path = json_dir / f"{stem}_top{args.top_k}_heads.json"

                overlay.save(overlay_path)
                np.save(map_path, sim_grid)

                top_json = []
                for rank, (r, c, s) in enumerate(peaks, start=1):
                    x0, y0, x1, y1 = patch_rect(r, c, patch_size)
                    cx = float((x0 + x1 + 1) / 2.0)
                    cy = float((y0 + y1 + 1) / 2.0)
                    top_json.append(
                        {
                            "rank": rank,
                            "patch_row": int(r),
                            "patch_col": int(c),
                            "score": float(s),
                            "patch_x0": int(x0),
                            "patch_y0": int(y0),
                            "patch_x1": int(x1),
                            "patch_y1": int(y1),
                            "patch_center_x": cx,
                            "patch_center_y": cy,
                            "patch_center_x_norm": cx / float(q_image.width),
                            "patch_center_y_norm": cy / float(q_image.height),
                        }
                    )

                vol_shape = load_volume_shape(args.segmented_root, animal, individual)
                payload = {
                    "animal": animal,
                    "individual": individual,
                    "angle": angle,
                    "query_image": str(query),
                    "overlay_image": str(overlay_path),
                    "similarity_map": str(map_path),
                    "image_width": int(q_image.width),
                    "image_height": int(q_image.height),
                    "grid_h": int(q_gh),
                    "grid_w": int(q_gw),
                    "volume_shape_xyz": vol_shape,
                    "mode": model_info["mode"],
                    "n_components_used": model_info["n_components"],
                    "patch_size": patch_size,
                    "prototype_count": model_info["prototype_count"],
                    "top_k": args.top_k,
                    "min_patch_distance": args.min_patch_distance,
                    "top_patches": top_json,
                }
                json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                total += 1
                log(f"    Angle {angle}: saved {overlay_path.name}")
            processed_individuals += 1

    log(f"Done. Saved results for {total} image(s).")


if __name__ == "__main__":
    main()
