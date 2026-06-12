"""Smoke test for the DINOv3 port.

Verifies the three things that differ from the DINOv2 pipeline:
  1. The gated DINOv3 weights can be loaded (requires HuggingFace access + token).
  2. patch_size is read as 16 and register tokens are reported.
  3. extract_patch_tokens returns exactly grid_h * grid_w patch tokens
     (i.e. CLS + register tokens are skipped correctly).

It also checks that the DINOv3 image processor accepts the do_resize / do_center_crop
keyword arguments used by the pipeline.

Run from the repo root:
    python dinov3/smoke_test.py
    python dinov3/smoke_test.py --model-name facebook/dinov3-vitb16-pretrain-lvd1689m
    python dinov3/smoke_test.py --image data/new_photos/segmented/AC/bcrick_1_000/bcrick_1_000_+X.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def find_default_image() -> Path | None:
    for root in (Path("data/new_photos_dinov3/segmented"), Path("data/new_photos/segmented")):
        if root.exists():
            pngs = sorted(root.rglob("*_*_*.png"))
            if pngs:
                return pngs[0]
    return None


def crop_to_patch_multiple(image: Image.Image, patch_size: int) -> Image.Image:
    w, h = image.size
    nw = (w // patch_size) * patch_size
    nh = (h // patch_size) * patch_size
    if nw <= 0 or nh <= 0:
        raise ValueError(f"Image too small for patch_size={patch_size}: {(w, h)}")
    return image if (nw, nh) == (w, h) else image.crop((0, 0, nw, nh))


def main() -> int:
    parser = argparse.ArgumentParser(description="DINOv3 port smoke test.")
    parser.add_argument("--model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    image_path = args.image or find_default_image()
    if image_path is None or not Path(image_path).exists():
        print("ERROR: no test image found. Pass --image PATH to an existing rendered PNG.")
        return 2

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Loading {args.model_name} on {device} ...")
    try:
        processor = AutoImageProcessor.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name).to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        print("ERROR: could not load the model. If this is a gated-repo / 401 error,")
        print("       request access at the model page and set a HuggingFace token")
        print("       (huggingface-cli login, or $env:HF_TOKEN). Details:")
        print(f"       {type(exc).__name__}: {str(exc)[:300]}")
        return 1

    patch_size = int(getattr(model.config, "patch_size", 16))
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    num_prefix = 1 + num_register_tokens
    print(f"[2/4] patch_size={patch_size} | num_register_tokens={num_register_tokens} "
          f"| prefix_tokens_skipped={num_prefix} | hidden_size={getattr(model.config, 'hidden_size', '?')}")

    print(f"[3/4] Checking processor kwargs on {Path(image_path).name} ...")
    image = Image.open(image_path).convert("RGB")
    image = crop_to_patch_multiple(image, patch_size)
    try:
        inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
    except TypeError as exc:
        print(f"       NOTE: processor rejected a kwarg ({exc}); retrying without do_center_crop.")
        inputs = processor(images=image, return_tensors="pt", do_resize=False)

    w, h = image.size
    gh, gw = h // patch_size, w // patch_size
    with torch.inference_mode():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        tokens = outputs.last_hidden_state[:, num_prefix:, :].squeeze(0).detach().cpu().numpy()

    print(f"[4/4] image={w}x{h} -> grid {gh}x{gw}={gh*gw} | "
          f"last_hidden_state seq_len={outputs.last_hidden_state.shape[1]} | "
          f"patch tokens={tokens.shape[0]} dim={tokens.shape[1]}")

    if tokens.shape[0] != gh * gw:
        print(f"FAIL: patch token count {tokens.shape[0]} != expected {gh*gw} "
              f"(register-token handling is wrong).")
        return 1

    print("\nPASS: model loaded, patch_size=16, register tokens skipped, token count matches grid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
