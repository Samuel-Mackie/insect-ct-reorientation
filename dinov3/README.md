# DINOv3-kopi af insect CT reorientation-pipelinen

Selvstændig kopi af pipelinen der bruger **DINOv3** (`facebook/dinov3-vits16/vitb16-pretrain-lvd1689m`)
i stedet for DINOv2. Symmetri-delen (`rotate_head_up_symmetry.py`, `symmetry.ipynb`) er **ikke** med.

Kør alle kommandoer **fra repo-roden** (ikke inde fra `dinov3/`), så de relative `data/`-stier passer.

## 1. Skaf adgang til DINOv3-vægtene (gated)

Vægtene kræver godkendt licens på HuggingFace:

1. Log ind på huggingface.co og åbn modelsiderne, klik **"Agree and access repository"**:
   - https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
   - https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
2. Lav en read-token under Settings → Access Tokens.
3. Autentificér (én af delene):
   ```powershell
   pip install huggingface_hub
   huggingface-cli login              # indsæt token
   # ELLER, kun for den aktuelle PowerShell-session:
   $env:HF_TOKEN = "hf_xxx"
   ```

## 2. Smoke test (verificér at DINOv3 virker)

```powershell
python dinov3/smoke_test.py
```
Bekræfter: model loader, `patch_size=16`, register-tokens springes over, og antallet af
patch-tokens matcher billed-grid'et. Forventet output slutter med `PASS:`.

## 3. Kør pipelinen

```powershell
python dinov3/run_pipeline.py            # fuld pipeline (small/vits16)
python dinov3/run_pipeline.py benchmark  # vits16 vs vitb16
```

Output skrives adskilt fra DINOv2-resultaterne:
```
data/new_photos_dinov3/{segmented,head_visualizations,head_top3,head_fused}
data/finished_photos_dinov3/{rotated,composite}
```

Begræns evt. scope på de enkelte scripts med `--animal AC` / `--max-files N`.

## Forskelle ift. DINOv2-versionen

| Sted | DINOv2 | DINOv3 |
|---|---|---|
| Model-id | `facebook/dinov2-{small,base}` | `facebook/dinov3-{vits16,vitb16}-pretrain-lvd1689m` |
| Patch size | 14 | **16** |
| Token-layout | `[CLS] + patches` → skip 1 | `[CLS] + N register + patches` → skip `1 + num_register_tokens` |
| Output-mapper | `data/new_photos/`, `data/finished_photos/` | `*_dinov3/` |

DINOv3 har register-tokens mellem CLS og patch-tokens. Patch-token-ekstraktionen læser
`model.config.num_register_tokens` dynamisk og springer `1 + num_register_tokens` tokens over.
Render-størrelsen (980 px) er ikke delelig med 16, så patch-grid-clipping bruger
`img // patch_size - 1` for at undgå en delvis kant-patch.
