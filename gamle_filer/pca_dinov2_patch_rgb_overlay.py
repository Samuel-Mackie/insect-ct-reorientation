from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch-wise DINOv2 PCA (RGB) with foreground filtering for one animal."
    )
    parser.add_argument("--animal", type=str, required=True, help="Animal code/folder, e.g. AC.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/new_photos/head_visualizations"),
        help="Root containing annotated images grouped by animal.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/new_photos/pca_dinov2_patch_rgb"),
        help="Root output folder.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-base",
        help="Hugging Face DINOv2 model id.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Blend amount for foreground only. 0=original, 1=PCA colors.",
    )
    parser.add_argument(
        "--bg-threshold",
        type=int,
        default=240,
        help="RGB threshold for white-background filtering.",
    )
    parser.add_argument(
        "--fg-patch-ratio",
        type=float,
        default=0.20,
        help="Minimum foreground pixel ratio in a patch to keep that patch.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Patch size in pixels for output visualization; use 14 to match previous setup.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cpu/cuda/cuda:0.",
    )
    return parser.parse_args()


def collect_images(animal_dir: Path) -> list[Path]:
    return sorted(animal_dir.rglob("*_annotated_heads.png"))


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
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    grid_h = h // patch_size
    grid_w = w // patch_size

    bg_mask = np.all(img_np > bg_threshold, axis=2)
    fg_mask_pixel = ~bg_mask
    fg_patch = np.zeros((grid_h, grid_w), dtype=bool)

    for i in range(grid_h):
        for j in range(grid_w):
            patch = fg_mask_pixel[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            fg_patch[i, j] = patch.mean() > fg_patch_ratio

    return fg_patch.reshape(-1), grid_h, grid_w


def infer_patches_for_image(
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

    fg_patch_flat, grid_h, grid_w = make_foreground_patch_mask(
        image=image,
        patch_size=patch_size,
        bg_threshold=bg_threshold,
        fg_patch_ratio=fg_patch_ratio,
    )
    if fg_patch_flat.sum() < 3:
        raise ValueError(f"Too few foreground patches in {image_path}")

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
            f"Patch count mismatch for {image_path}: got {patch_tokens.shape[0]}, expected {expected}"
        )

    return {
        "path": image_path,
        "image": image,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "patch_tokens": patch_tokens,
        "fg_patch_flat": fg_patch_flat,
    }


def normalize_with_percentiles(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    scale = np.maximum(hi - lo, 1e-6)
    rgb = (values - lo) / scale
    return np.clip(rgb, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    animal = args.animal.strip()
    alpha = float(np.clip(args.alpha, 0.0, 1.0))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    model_patch = int(getattr(model.config, "patch_size", 14))
    if model_patch != args.patch_size:
        raise ValueError(
            f"Patch size mismatch: model patch_size={model_patch}, requested --patch-size={args.patch_size}"
        )

    animal_dir = args.input_root / animal
    if not animal_dir.exists():
        raise FileNotFoundError(f"Animal folder not found: {animal_dir}")

    image_paths = collect_images(animal_dir)
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images for PCA, found {len(image_paths)} in {animal_dir}")

    print(f"Animal: {animal}")
    print(f"Images: {len(image_paths)}")
    print(f"Model: {args.model_name}")
    print(f"Patch size: {args.patch_size}")

    infos: list[dict[str, object]] = []
    all_fg_tokens: list[np.ndarray] = []
    for p in image_paths:
        info = infer_patches_for_image(
            image_path=p,
            processor=processor,
            model=model,
            patch_size=args.patch_size,
            bg_threshold=args.bg_threshold,
            fg_patch_ratio=args.fg_patch_ratio,
            device=device,
        )
        infos.append(info)
        tokens = info["patch_tokens"]  # type: ignore[assignment]
        fg = info["fg_patch_flat"]  # type: ignore[assignment]
        all_fg_tokens.append(tokens[fg])  # type: ignore[index]

    fg_tokens_matrix = np.concatenate(all_fg_tokens, axis=0)
    pca = PCA(n_components=3)
    pca.fit(fg_tokens_matrix)

    fg_pc = pca.transform(fg_tokens_matrix)
    lo = np.percentile(fg_pc, 1, axis=0)
    hi = np.percentile(fg_pc, 99, axis=0)

    out_dir = args.output_root / animal
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "animal": animal,
        "model_name": args.model_name,
        "patch_size": args.patch_size,
        "n_images": len(infos),
        "n_foreground_patches_total": int(fg_tokens_matrix.shape[0]),
        "feature_dim": int(fg_tokens_matrix.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc_norm_percentile_1": lo.tolist(),
        "pc_norm_percentile_99": hi.tolist(),
        "alpha": alpha,
        "bg_threshold": args.bg_threshold,
        "fg_patch_ratio": args.fg_patch_ratio,
        "images": [],
    }

    for info in infos:
        path: Path = info["path"]  # type: ignore[assignment]
        image: Image.Image = info["image"]  # type: ignore[assignment]
        grid_h = int(info["grid_h"])  # type: ignore[arg-type]
        grid_w = int(info["grid_w"])  # type: ignore[arg-type]
        tokens = info["patch_tokens"]  # type: ignore[assignment]
        fg_flat = info["fg_patch_flat"]  # type: ignore[assignment]

        pc = pca.transform(tokens)
        rgb = normalize_with_percentiles(pc, lo=lo, hi=hi)

        patch_rgb = np.full((grid_h * grid_w, 3), 1.0, dtype=np.float32)
        patch_rgb[fg_flat] = rgb[fg_flat]
        patch_rgb_img = (patch_rgb.reshape(grid_h, grid_w, 3) * 255.0).astype(np.uint8)

        # Convert patch grid to full image with exact 14x14 (or chosen) patch cells.
        patch_rgb_full = np.repeat(
            np.repeat(patch_rgb_img, args.patch_size, axis=0),
            args.patch_size,
            axis=1,
        )
        patch_rgb_full = patch_rgb_full[: image.height, : image.width, :]

        orig = np.array(image.convert("RGB"))
        overlay = np.full_like(orig, 255, dtype=np.uint8)

        fg_pixel = np.repeat(
            np.repeat(fg_flat.reshape(grid_h, grid_w), args.patch_size, axis=0),
            args.patch_size,
            axis=1,
        )
        fg_pixel = fg_pixel[: image.height, : image.width]

        blend = (1.0 - alpha) * orig.astype(np.float32) + alpha * patch_rgb_full.astype(np.float32)
        overlay[fg_pixel] = np.clip(blend[fg_pixel], 0, 255).astype(np.uint8)

        rel = path.relative_to(animal_dir)
        save_dir = out_dir / rel.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem.replace("_annotated_heads", "")
        overlay_path = save_dir / f"{stem}_pca_rgb_overlay_fg.png"
        Image.fromarray(overlay).save(overlay_path)

        summary["images"].append(
            {
                "input_image": str(path),
                "overlay_image": str(overlay_path),
                "grid_h": grid_h,
                "grid_w": grid_w,
                "image_w": image.width,
                "image_h": image.height,
                "foreground_patches": int(fg_flat.sum()),
                "total_patches": int(fg_flat.size),
            }
        )

    (out_dir / "pca_patch_rgb_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    evr = pca.explained_variance_ratio_
    print(f"Saved overlays to: {out_dir}")
    print(f"Explained variance ratio: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}, PC3={evr[2]:.4f}")


if __name__ == "__main__":
    main()
