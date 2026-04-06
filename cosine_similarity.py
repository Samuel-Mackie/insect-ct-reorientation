from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Head-patch prototype matching with cosine similarity for one or many animal/angle combinations."
    )
    parser.add_argument(
        "--animal",
        type=str,
        default=None,
        help="Animal code, e.g. AC. If omitted, all animals under head-vis-root are processed.",
    )
    parser.add_argument(
        "--angle",
        type=str,
        default=None,
        help="One angle (+X,-X,+Y,-Y,+Z,-Z). If omitted, all angles are processed.",
    )
    parser.add_argument(
        "--query-image",
        type=Path,
        default=None,
        help="Optional query image path. If omitted, the first annotated image for this angle is used.",
    )
    parser.add_argument(
        "--head-vis-root",
        type=Path,
        default=Path("data/new_photos/head_visualizations"),
        help="Root containing head_projection.json per scan.",
    )
    parser.add_argument(
        "--segmented-root",
        type=Path,
        default=Path("data/new_photos/segmented"),
        help="Root containing segmented view images used for DINO tokens.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/new_photos/cosine_similarity"),
        help="Root folder for overlay outputs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-base",
        help="Hugging Face DINO model id.",
    )
    parser.add_argument(
        "--explained-variance",
        type=float,
        default=None,
        help="If set in (0,1), use PCA components up to this cumulative variance. If omitted, use all components.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.60,
        help="Overlay blend amount. 0=original image only, 1=similarity colormap only.",
    )
    parser.add_argument(
        "--bg-threshold",
        type=int,
        default=240,
        help="White-background threshold for foreground masking.",
    )
    parser.add_argument(
        "--fg-patch-ratio",
        type=float,
        default=0.20,
        help="Minimum foreground pixel ratio in a patch to be treated as foreground.",
    )
    parser.add_argument(
        "--exclude-query-from-prototype",
        action="store_true",
        help="Exclude query image from prototype averaging if it is part of the annotated set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cpu/cuda/cuda:0.",
    )
    return parser.parse_args()


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
        views = data.get("views", {})
        if angle not in views:
            continue

        view = views[angle]
        scan_stem = meta_path.parent.name
        segmented_img = segmented_root / animal / scan_stem / f"{scan_stem}_{idx}_{angle}.png"
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
    new_w = (w // patch_size) * patch_size
    new_h = (h // patch_size) * patch_size
    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"Image too small for patch_size={patch_size}: {(w, h)}")
    if new_w == w and new_h == h:
        return image
    return image.crop((0, 0, new_w, new_h))


def make_foreground_patch_mask(
    image: Image.Image,
    patch_size: int,
    bg_threshold: int,
    fg_patch_ratio: float,
) -> tuple[np.ndarray, int, int]:
    img = np.array(image)
    h, w = img.shape[:2]
    grid_h = h // patch_size
    grid_w = w // patch_size

    bg_pixel = np.all(img > bg_threshold, axis=2)
    fg_pixel = ~bg_pixel

    fg_patch = np.zeros((grid_h, grid_w), dtype=bool)
    for i in range(grid_h):
        for j in range(grid_w):
            patch = fg_pixel[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            fg_patch[i, j] = patch.mean() > fg_patch_ratio
    return fg_patch.reshape(-1), grid_h, grid_w


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
    image = crop_to_patch_multiple(image, patch_size=patch_size)

    fg_flat, grid_h, grid_w = make_foreground_patch_mask(
        image=image,
        patch_size=patch_size,
        bg_threshold=bg_threshold,
        fg_patch_ratio=fg_patch_ratio,
    )

    with torch.inference_mode():
        inputs = processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        patch_tokens = outputs.last_hidden_state[:, 1:, :].squeeze(0).detach().cpu().numpy()

    expected = grid_h * grid_w
    if patch_tokens.shape[0] != expected:
        raise ValueError(
            f"Patch count mismatch in {image_path}: got {patch_tokens.shape[0]}, expected {expected}"
        )
    return {
        "image": image,
        "tokens": patch_tokens,
        "fg_flat": fg_flat,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return x / denom


def choose_components_by_variance(all_tokens: np.ndarray, threshold: float) -> tuple[PCA, int]:
    pca_full = PCA(n_components=min(all_tokens.shape[0], all_tokens.shape[1]))
    pca_full.fit(all_tokens)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, threshold, side="left") + 1)
    n_comp = max(1, min(n_comp, all_tokens.shape[1]))

    pca = PCA(n_components=n_comp)
    pca.fit(all_tokens)
    return pca, n_comp


def similarity_overlay(
    image: Image.Image,
    sim_grid: np.ndarray,
    fg_patch_mask: np.ndarray,
    patch_size: int,
    alpha: float,
) -> Image.Image:
    grid_h, grid_w = sim_grid.shape
    sim = sim_grid.copy()

    fg = fg_patch_mask.reshape(grid_h, grid_w)
    if fg.any():
        vals = sim[fg]
        lo, hi = np.percentile(vals, [2, 98])
        sim = (sim - lo) / max(hi - lo, 1e-6)
    else:
        sim[:] = 0.0
    sim = np.clip(sim, 0.0, 1.0)

    cmap = plt.get_cmap("turbo")
    heat_small = (cmap(sim)[..., :3] * 255).astype(np.uint8)
    heat = np.repeat(np.repeat(heat_small, patch_size, axis=0), patch_size, axis=1)
    heat = heat[: image.height, : image.width, :]

    orig = np.array(image.convert("RGB"))
    overlay = np.full_like(orig, 255, dtype=np.uint8)

    fg_pixel = np.repeat(np.repeat(fg, patch_size, axis=0), patch_size, axis=1)
    fg_pixel = fg_pixel[: image.height, : image.width]

    blend = (1 - alpha) * orig.astype(np.float32) + alpha * heat.astype(np.float32)
    overlay[fg_pixel] = np.clip(blend[fg_pixel], 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def discover_animals(head_vis_root: Path) -> list[str]:
    if not head_vis_root.exists():
        raise FileNotFoundError(f"Head visualization root not found: {head_vis_root}")
    return sorted(p.name for p in head_vis_root.iterdir() if p.is_dir())


def discover_individuals(head_vis_root: Path, animal: str) -> list[str]:
    animal_dir = head_vis_root / animal
    if not animal_dir.exists():
        return []
    out: list[str] = []
    for p in sorted(animal_dir.iterdir()):
        if p.is_dir() and (p / "head_projection.json").exists():
            out.append(p.name)
    return out


def discover_segmented_individuals(segmented_root: Path, animal: str) -> list[str]:
    species_dir = segmented_root / animal
    if not species_dir.exists():
        return []
    return sorted(p.name for p in species_dir.iterdir() if p.is_dir())


def normalize_angle(angle: str) -> str:
    a = angle.strip()
    if a not in ANGLE_TO_INDEX:
        raise ValueError(f"Invalid angle '{angle}'. Valid: {list(ANGLE_TO_INDEX.keys())}")
    return a


def process_combo(
    *,
    args: argparse.Namespace,
    processor: AutoImageProcessor,
    model: AutoModel,
    patch_size: int,
    device: str,
    animal: str,
    angle: str,
    token_cache: dict[Path, dict[str, object]],
    query_paths_override: list[Path] | None = None,
) -> int:
    annotated = load_head_annotations(
        head_vis_root=args.head_vis_root,
        animal=animal,
        angle=angle,
        segmented_root=args.segmented_root,
    )
    if not annotated:
        log(f"      [{animal} {angle}] no annotations available for prototype")
        return 0

    by_path = {Path(e["segmented_image"]): e for e in annotated}  # type: ignore[index]

    missing_proto_paths = [p for p in by_path if p not in token_cache]
    if missing_proto_paths:
        log(f"      [{animal} {angle}] extracting prototype tokens: {len(missing_proto_paths)} image(s)")
    for i, path in enumerate(missing_proto_paths, start=1):
        log(f"      [{animal} {angle}] prototype token {i}/{len(missing_proto_paths)}: {path.name}")
        token_cache[path] = extract_patch_tokens(
            image_path=path,
            processor=processor,
            model=model,
            patch_size=patch_size,
            bg_threshold=args.bg_threshold,
            fg_patch_ratio=args.fg_patch_ratio,
            device=device,
        )

    if query_paths_override is not None:
        query_paths = query_paths_override
        missing_query_paths = [qp for qp in query_paths if qp not in token_cache and qp.exists()]
        if missing_query_paths:
            log(f"      [{animal} {angle}] extracting query tokens: {len(missing_query_paths)} image(s)")
        for i, qp in enumerate(missing_query_paths, start=1):
            log(f"      [{animal} {angle}] query token {i}/{len(missing_query_paths)}: {qp.name}")
            token_cache[qp] = extract_patch_tokens(
                image_path=qp,
                processor=processor,
                model=model,
                patch_size=patch_size,
                bg_threshold=args.bg_threshold,
                fg_patch_ratio=args.fg_patch_ratio,
                device=device,
            )
    elif args.query_image is not None:
        query_paths = [args.query_image]
        if query_paths[0] not in token_cache:
            log(f"      [{animal} {angle}] extracting query token: {query_paths[0].name}")
            token_cache[query_paths[0]] = extract_patch_tokens(
                image_path=query_paths[0],
                processor=processor,
                model=model,
                patch_size=patch_size,
                bg_threshold=args.bg_threshold,
                fg_patch_ratio=args.fg_patch_ratio,
                device=device,
            )
    else:
        query_paths = sorted(by_path.keys())

    pca_fit_list: list[np.ndarray] = []
    for path, entry in by_path.items():
        feats = token_cache[path]
        tokens = feats["tokens"]  # type: ignore[assignment]
        fg_flat = feats["fg_flat"]  # type: ignore[assignment]
        row = int(entry["patch_row"])  # type: ignore[arg-type]
        col = int(entry["patch_col"])  # type: ignore[arg-type]
        grid_w = int(feats["grid_w"])  # type: ignore[arg-type]
        grid_h = int(feats["grid_h"])  # type: ignore[arg-type]
        if not (0 <= row < grid_h and 0 <= col < grid_w):
            continue
        pca_fit_list.append(tokens[fg_flat])  # type: ignore[index]

    if not pca_fit_list:
        return 0

    mode = "all_components"
    n_components = int(token_cache[next(iter(by_path.keys()))]["tokens"].shape[1])  # type: ignore[index]
    pca = None
    if args.explained_variance is not None:
        if not (0.0 < args.explained_variance <= 1.0):
            raise ValueError("--explained-variance must be in (0, 1].")
        if args.explained_variance < 1.0:
            fit_matrix = np.concatenate(pca_fit_list, axis=0)
            log(f"      [{animal} {angle}] fitting PCA for explained variance {args.explained_variance:.3f}")
            pca, n_components = choose_components_by_variance(fit_matrix, args.explained_variance)
            mode = f"pca_{args.explained_variance:.3f}"

    alpha = float(np.clip(args.alpha, 0.0, 1.0))

    written = 0
    log(f"      [{animal} {angle}] computing cosine overlays for {len(query_paths)} query image(s)")
    for query_path in query_paths:
        if not query_path.exists():
            continue

        proto_list: list[np.ndarray] = []
        for path, entry in by_path.items():
            if args.exclude_query_from_prototype and path.resolve() == query_path.resolve():
                continue
            feats = token_cache[path]
            tokens = feats["tokens"]  # type: ignore[assignment]
            row = int(entry["patch_row"])  # type: ignore[arg-type]
            col = int(entry["patch_col"])  # type: ignore[arg-type]
            grid_w = int(feats["grid_w"])  # type: ignore[arg-type]
            grid_h = int(feats["grid_h"])  # type: ignore[arg-type]
            if not (0 <= row < grid_h and 0 <= col < grid_w):
                continue
            idx = row * grid_w + col
            proto_list.append(tokens[idx])

        if not proto_list:
            continue

        prototype = np.mean(np.stack(proto_list, axis=0), axis=0, keepdims=True)
        q = token_cache[query_path]
        q_tokens = q["tokens"]  # type: ignore[assignment]
        q_fg = q["fg_flat"]  # type: ignore[assignment]
        q_grid_h = int(q["grid_h"])  # type: ignore[arg-type]
        q_grid_w = int(q["grid_w"])  # type: ignore[arg-type]
        q_image: Image.Image = q["image"]  # type: ignore[assignment]

        if pca is not None:
            prototype_m = pca.transform(prototype)
            q_tokens_m = pca.transform(q_tokens)
        else:
            prototype_m = prototype
            q_tokens_m = q_tokens

        prototype_m = l2_normalize(prototype_m, axis=1)
        q_tokens_m = l2_normalize(q_tokens_m, axis=1)
        sims = (q_tokens_m @ prototype_m.T).squeeze(1)
        sim_grid = sims.reshape(q_grid_h, q_grid_w)

        overlay = similarity_overlay(
            image=q_image,
            sim_grid=sim_grid,
            fg_patch_mask=q_fg,
            patch_size=patch_size,
            alpha=alpha,
        )

        query_stem = query_path.stem
        if query_path in by_path:
            individual = str(by_path[query_path]["scan_stem"])
        else:
            individual = query_path.parent.name

        base_dir = args.output_root / animal / individual
        viz_dir = base_dir / "visualisations"
        data_dir = base_dir / "data"
        json_dir = base_dir / "json"
        viz_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = viz_dir / f"{query_stem}_{angle}_cosine_overlay.png"
        overlay.save(overlay_path)
        np.save(data_dir / f"{query_stem}_{angle}_cosine_patch_map.npy", sim_grid)
        result = {
            "animal": animal,
            "individual": individual,
            "angle": angle,
            "query_image": str(query_path),
            "overlay_image": str(overlay_path),
            "model_name": args.model_name,
            "patch_size": patch_size,
            "prototype_count": len(proto_list),
            "mode": mode,
            "n_components_used": n_components,
            "exclude_query_from_prototype": args.exclude_query_from_prototype,
            "explained_variance_input": args.explained_variance,
            "similarity_min": float(sims.min()),
            "similarity_max": float(sims.max()),
            "similarity_mean": float(sims.mean()),
        }
        (json_dir / f"{query_stem}_{angle}_run.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        written += 1
        log(f"      [{animal} {angle}] saved overlay for {query_stem}")

    return written


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    patch_size = int(getattr(model.config, "patch_size", 14))

    animals = [args.animal.strip()] if args.animal else discover_animals(args.head_vis_root)
    if args.angle:
        angles = [normalize_angle(args.angle)]
    else:
        angles = list(ANGLE_TO_INDEX.keys())

    if args.query_image is not None and (len(animals) != 1 or len(angles) != 1):
        raise ValueError("--query-image can only be used when exactly one animal and one angle are selected.")

    if args.query_image is not None and not args.query_image.exists():
        raise FileNotFoundError(f"Query image not found: {args.query_image}")

    token_cache: dict[Path, dict[str, object]] = {}
    total_written = 0
    for animal_idx, animal in enumerate(animals, start=1):
        log(f"Animal {animal_idx}/{len(animals)}: {animal}")
        animal_total = 0

        if args.query_image is not None:
            angle = angles[0]
            written = process_combo(
                args=args,
                processor=processor,
                model=model,
                patch_size=patch_size,
                device=device,
                animal=animal,
                angle=angle,
                token_cache=token_cache,
                query_paths_override=[args.query_image],
            )
            log(f"  Query mode {angle}: {written} overlay(s)")
            animal_total += written
            total_written += written
            log(f"Animal {animal}: total {animal_total} overlay(s)")
            continue

        individuals = discover_segmented_individuals(args.segmented_root, animal)
        log(f"  Found {len(individuals)} individuals in segmented data")
        for indiv_idx, individual in enumerate(individuals, start=1):
            log(f"  Individual {indiv_idx}/{len(individuals)}: {individual}")
            indiv_total = 0
            for angle in angles:
                query_path = args.segmented_root / animal / individual / f"{individual}_{ANGLE_TO_INDEX[angle]}_{angle}.png"
                if not query_path.exists():
                    log(f"    Angle {angle}: 0 overlay(s) (missing query image)")
                    continue
                log(f"    Angle {angle}: running...")
                written = process_combo(
                    args=args,
                    processor=processor,
                    model=model,
                    patch_size=patch_size,
                    device=device,
                    animal=animal,
                    angle=angle,
                    token_cache=token_cache,
                    query_paths_override=[query_path],
                )
                log(f"    Angle {angle}: {written} overlay(s)")
                indiv_total += written
                animal_total += written
                total_written += written
            log(f"  Individual {individual}: total {indiv_total} overlay(s)")
        log(f"Animal {animal}: total {animal_total} overlay(s)")

    log(f"Done. Total overlays written: {total_written}")


if __name__ == "__main__":
    main()
