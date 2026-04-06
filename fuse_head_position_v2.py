from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ANGLE_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse head position from 6 views using camera-ray triangulation (v2)."
    )
    parser.add_argument("--animal", type=str, default=None, help="Species code (e.g. AC). Omit for all.")
    parser.add_argument(
        "--top3-root",
        type=Path,
        default=Path("data/new_photos/head_top3"),
        help="Root containing top3_head_patches outputs.",
    )
    parser.add_argument(
        "--segmented-root",
        type=Path,
        default=Path("data/new_photos/segmented"),
        help="Root containing segmentation metadata.json files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/new_photos/head_fused_v2"),
        help="Root for fused outputs.",
    )
    parser.add_argument(
        "--score-temperature",
        type=float,
        default=8.0,
        help="Softmax-like score temperature for view weights.",
    )
    parser.add_argument(
        "--camera-fov-deg",
        type=float,
        default=30.0,
        help="Vertical camera FOV in degrees used to build rays (vtk default ~30).",
    )
    parser.add_argument(
        "--top-k-per-view",
        type=int,
        default=3,
        help="Use top-k patch candidates per angle when building rays.",
    )
    parser.add_argument(
        "--robust-iters",
        type=int,
        default=5,
        help="IRLS iterations for robust triangulation.",
    )
    parser.add_argument(
        "--huber-k",
        type=float,
        default=1.5,
        help="Huber threshold multiplier for robust reweighting.",
    )
    parser.add_argument(
        "--rank-decay",
        type=float,
        default=0.75,
        help="Additional decay for lower-ranked candidates: 1/(rank^rank_decay).",
    )
    return parser.parse_args()


def discover_animals(top3_root: Path) -> list[str]:
    if not top3_root.exists():
        return []
    return sorted(p.name for p in top3_root.iterdir() if p.is_dir())


def discover_individuals(top3_root: Path, animal: str) -> list[str]:
    species_dir = top3_root / animal
    if not species_dir.exists():
        return []
    return sorted(p.name for p in species_dir.iterdir() if p.is_dir())


def latest_json_by_angle(json_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in json_dir.glob("*_top*_heads.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            angle = d.get("angle")
            if angle not in ANGLE_NAMES:
                continue
            if angle not in out or p.stat().st_mtime > out[angle].stat().st_mtime:
                out[angle] = p
        except Exception:
            continue
    return out


def score_to_weight(score: float, temperature: float) -> float:
    return float(np.exp(np.clip(score * temperature, -40.0, 40.0)))


def load_camera_by_angle(segmented_meta: Path) -> dict[str, dict[str, object]]:
    d = json.loads(segmented_meta.read_text(encoding="utf-8"))
    views = d.get("render_geometry", {}).get("camera_views", [])
    out: dict[str, dict[str, object]] = {}
    for v in views:
        angle = v.get("angle")
        if angle in ANGLE_NAMES:
            out[angle] = v
    return out


def camera_basis(cam_pos: np.ndarray, focal: np.ndarray, view_up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = focal - cam_pos
    f = f / max(np.linalg.norm(f), 1e-12)
    u0 = view_up / max(np.linalg.norm(view_up), 1e-12)
    r = np.cross(f, u0)
    r = r / max(np.linalg.norm(r), 1e-12)
    u = np.cross(r, f)
    u = u / max(np.linalg.norm(u), 1e-12)
    return r, u, f


def build_ray_from_pixel(
    nx: float,
    ny: float,
    image_w: int,
    image_h: int,
    cam_pos: np.ndarray,
    focal: np.ndarray,
    view_up: np.ndarray,
    fov_y_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Normalized display -> NDC
    x_ndc = (nx - 0.5) * 2.0
    y_ndc = (0.5 - ny) * 2.0
    aspect = float(image_w) / max(float(image_h), 1.0)
    tan_half = np.tan(np.deg2rad(fov_y_deg) * 0.5)

    # Camera-local direction (right, up, forward)
    d_cam = np.array([x_ndc * aspect * tan_half, y_ndc * tan_half, 1.0], dtype=float)
    d_cam = d_cam / max(np.linalg.norm(d_cam), 1e-12)

    r, u, f = camera_basis(cam_pos, focal, view_up)
    d_world = d_cam[0] * r + d_cam[1] * u + d_cam[2] * f
    d_world = d_world / max(np.linalg.norm(d_world), 1e-12)
    return cam_pos, d_world


def triangulate_rays_weighted(origins: list[np.ndarray], dirs: list[np.ndarray], weights: list[float]) -> tuple[np.ndarray, float]:
    # Solve: sum w(I - dd^T) p = sum w(I - dd^T) o
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    for o, d, w in zip(origins, dirs, weights):
        d = d / max(np.linalg.norm(d), 1e-12)
        M = np.eye(3) - np.outer(d, d)
        A += w * M
        b += w * (M @ o)
    p = np.linalg.solve(A + np.eye(3) * 1e-9, b)

    dists = []
    for o, d in zip(origins, dirs):
        d = d / max(np.linalg.norm(d), 1e-12)
        dists.append(float(np.linalg.norm(np.cross(p - o, d))))
    return p, float(np.mean(dists))


def robust_triangulate(
    origins: list[np.ndarray],
    dirs: list[np.ndarray],
    base_weights: list[float],
    iters: int,
    huber_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    o = np.stack(origins, axis=0)
    d = np.stack(dirs, axis=0)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    w_base = np.array(base_weights, dtype=float)
    w = w_base.copy()

    p = np.zeros(3, dtype=float)
    for _ in range(max(iters, 1)):
        p, _ = triangulate_rays_weighted(list(o), list(d), list(w))
        residuals = np.linalg.norm(np.cross(p[None, :] - o, d), axis=1)
        med = float(np.median(residuals))
        sigma = max(med * 1.4826, 1e-6)
        delta = huber_k * sigma
        rw = np.ones_like(residuals)
        big = residuals > delta
        rw[big] = delta / np.maximum(residuals[big], 1e-12)
        w = w_base * rw

    residuals = np.linalg.norm(np.cross(p[None, :] - o, d), axis=1)
    return p, residuals, w


def project_point_to_view(
    p: np.ndarray,
    image_w: int,
    image_h: int,
    cam_pos: np.ndarray,
    focal: np.ndarray,
    view_up: np.ndarray,
    fov_y_deg: float,
) -> tuple[float, float]:
    r, u, f = camera_basis(cam_pos, focal, view_up)
    v = p - cam_pos
    x = float(np.dot(v, r))
    y = float(np.dot(v, u))
    z = float(np.dot(v, f))
    if z <= 1e-9:
        return 0.5, 0.5
    aspect = float(image_w) / max(float(image_h), 1.0)
    tan_half = np.tan(np.deg2rad(fov_y_deg) * 0.5)
    x_ndc = x / (z * tan_half * aspect)
    y_ndc = y / (z * tan_half)
    nx = 0.5 + 0.5 * x_ndc
    ny = 0.5 - 0.5 * y_ndc
    return float(nx), float(ny)


def draw_overlay(
    query_image: Path,
    out_path: Path,
    pred_nx: float,
    pred_ny: float,
    obs_nx: float,
    obs_ny: float,
) -> None:
    img = Image.open(query_image).convert("RGB")
    w, h = img.size
    px, py = pred_nx * w, pred_ny * h
    ox, oy = obs_nx * w, obs_ny * h
    draw = ImageDraw.Draw(img)
    draw.line((px - 10, py, px + 10, py), fill=(0, 180, 0), width=3)
    draw.line((px, py - 10, px, py + 10), fill=(0, 180, 0), width=3)
    draw.ellipse((ox - 6, oy - 6, ox + 6, oy + 6), outline=(220, 0, 0), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    args = parse_args()
    animals = [args.animal] if args.animal else discover_animals(args.top3_root)
    if not animals:
        log("No animals found in top3 root.")
        return

    total = 0
    for ai, animal in enumerate(animals, start=1):
        log(f"Animal {ai}/{len(animals)}: {animal}")
        individuals = discover_individuals(args.top3_root, animal)
        for ii, individual in enumerate(individuals, start=1):
            log(f"  Individual {ii}/{len(individuals)}: {individual}")
            json_dir = args.top3_root / animal / individual / "json"
            angle_to_json = latest_json_by_angle(json_dir)
            if not angle_to_json:
                log("    No top3 files, skipping")
                continue

            seg_meta = args.segmented_root / animal / individual / "metadata.json"
            if not seg_meta.exists():
                log("    Missing segmented metadata.json, skipping")
                continue
            cam_by_angle = load_camera_by_angle(seg_meta)
            if not cam_by_angle:
                log("    Missing camera metadata, skipping")
                continue

            origins: list[np.ndarray] = []
            dirs: list[np.ndarray] = []
            weights: list[float] = []
            ray_entries: list[dict[str, object]] = []
            volume_shape_xyz = None

            for angle, jp in sorted(angle_to_json.items()):
                if angle not in cam_by_angle:
                    continue
                d = json.loads(jp.read_text(encoding="utf-8"))
                if volume_shape_xyz is None:
                    volume_shape_xyz = d.get("volume_shape_xyz")
                tops = d.get("top_patches", [])
                if not tops:
                    continue

                cam = cam_by_angle[angle]
                cam_pos = np.array(cam["camera_position"], dtype=float)
                focal = np.array(cam["camera_focal_point"], dtype=float)
                view_up = np.array(cam["view_up"], dtype=float)
                image_w = int(d.get("image_width", 980))
                image_h = int(d.get("image_height", 980))
                for t in tops[: max(1, args.top_k_per_view)]:
                    rank = int(t.get("rank", 1))
                    nx = float(t["patch_center_x_norm"])
                    ny = float(t["patch_center_y_norm"])
                    score = float(t["score"])
                    w = score_to_weight(score, args.score_temperature) / (rank ** args.rank_decay)

                    o, dd = build_ray_from_pixel(
                        nx=nx,
                        ny=ny,
                        image_w=image_w,
                        image_h=image_h,
                        cam_pos=cam_pos,
                        focal=focal,
                        view_up=view_up,
                        fov_y_deg=args.camera_fov_deg,
                    )
                    origins.append(o)
                    dirs.append(dd)
                    weights.append(w)
                    ray_entries.append(
                        {
                            "angle": angle,
                            "json_file": str(jp),
                            "query_image": d.get("query_image"),
                            "patch": t,
                            "weight_base": w,
                            "camera_position": cam["camera_position"],
                            "camera_focal_point": cam["camera_focal_point"],
                            "view_up": cam["view_up"],
                            "image_width": image_w,
                            "image_height": image_h,
                        }
                    )

            if len(origins) < 2:
                log("    Too few usable angles for triangulation, skipping")
                continue

            p_xyz, ray_residuals, w_final = robust_triangulate(
                origins=origins,
                dirs=dirs,
                base_weights=weights,
                iters=args.robust_iters,
                huber_k=args.huber_k,
            )
            mean_ray_dist = float(np.mean(ray_residuals))

            if isinstance(volume_shape_xyz, list) and len(volume_shape_xyz) == 3:
                sx, sy, sz = [int(v) for v in volume_shape_xyz]
                px = float(np.clip(p_xyz[0], 0, max(sx - 1, 1)))
                py = float(np.clip(p_xyz[1], 0, max(sy - 1, 1)))
                pz = float(np.clip(p_xyz[2], 0, max(sz - 1, 1)))
                p_clip = np.array([px, py, pz], dtype=float)
                norm_xyz = {"x": px / max(sx - 1, 1), "y": py / max(sy - 1, 1), "z": pz / max(sz - 1, 1)}
                voxel_xyz = [px, py, pz]
            else:
                p_clip = p_xyz.copy()
                norm_xyz = {"x": float(p_clip[0]), "y": float(p_clip[1]), "z": float(p_clip[2])}
                voxel_xyz = [float(p_clip[0]), float(p_clip[1]), float(p_clip[2])]

            # Select best representative candidate per angle by reprojection error to fused point.
            per_angle_best: dict[str, dict[str, object]] = {}
            for ent in ray_entries:
                angle = str(ent["angle"])
                cam_pos = np.array(ent["camera_position"], dtype=float)
                focal = np.array(ent["camera_focal_point"], dtype=float)
                view_up = np.array(ent["view_up"], dtype=float)
                image_w = int(ent["image_width"])
                image_h = int(ent["image_height"])
                pred_nx, pred_ny = project_point_to_view(
                    p=p_clip,
                    image_w=image_w,
                    image_h=image_h,
                    cam_pos=cam_pos,
                    focal=focal,
                    view_up=view_up,
                    fov_y_deg=args.camera_fov_deg,
                )
                t = ent["patch"]
                obs_nx = float(t["patch_center_x_norm"])
                obs_ny = float(t["patch_center_y_norm"])
                err = float(np.hypot(pred_nx - obs_nx, pred_ny - obs_ny))
                cand = {
                    "json_file": ent["json_file"],
                    "query_image": ent["query_image"],
                    "top_patch": t,
                    "weight_base": float(ent["weight_base"]),
                    "pred_patch_center_x_norm": pred_nx,
                    "pred_patch_center_y_norm": pred_ny,
                    "obs_patch_center_x_norm": obs_nx,
                    "obs_patch_center_y_norm": obs_ny,
                    "reprojection_error_norm": err,
                }
                if angle not in per_angle_best or err < float(per_angle_best[angle]["reprojection_error_norm"]):
                    per_angle_best[angle] = cand

            reproj = {}
            for angle, cand in per_angle_best.items():
                reproj[angle] = {
                    "pred_patch_center_x_norm": float(cand["pred_patch_center_x_norm"]),
                    "pred_patch_center_y_norm": float(cand["pred_patch_center_y_norm"]),
                    "obs_patch_center_x_norm": float(cand["obs_patch_center_x_norm"]),
                    "obs_patch_center_y_norm": float(cand["obs_patch_center_y_norm"]),
                    "reprojection_error_norm": float(cand["reprojection_error_norm"]),
                }

            per_angle_top = {}
            for angle, cand in per_angle_best.items():
                per_angle_top[angle] = {
                    "json_file": str(cand["json_file"]),
                    "query_image": str(cand["query_image"]),
                    "top1_patch": cand["top_patch"],
                    "weight": float(cand["weight_base"]),
                }

            mean_reproj = float(np.mean([v["reprojection_error_norm"] for v in reproj.values()])) if reproj else 1.0
            conf = float(np.clip(1.0 - (mean_reproj * 3.0 + mean_ray_dist / 120.0), 0.0, 1.0))

            base_dir = args.output_root / animal / individual
            viz_dir = base_dir / "visualisations"
            data_dir = base_dir / "data"
            json_out_dir = base_dir / "json"
            viz_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            json_out_dir.mkdir(parents=True, exist_ok=True)

            result = {
                "animal": animal,
                "individual": individual,
                "mode": "camera_ray_triangulation_v2",
                "camera_fov_deg": args.camera_fov_deg,
                "top_k_per_view": args.top_k_per_view,
                "robust_iters": args.robust_iters,
                "huber_k": args.huber_k,
                "rank_decay": args.rank_decay,
                "used_angles": sorted(list(per_angle_top.keys())),
                "num_rays_total": len(ray_entries),
                "estimated_head_xyz_norm": norm_xyz,
                "estimated_head_xyz_voxel": voxel_xyz,
                "mean_ray_distance_voxels": mean_ray_dist,
                "mean_reprojection_error_norm": mean_reproj,
                "confidence": conf,
                "score_temperature": args.score_temperature,
                "per_angle_top1": per_angle_top,
                "per_angle_reprojection": reproj,
                "ray_residuals_summary": {
                    "min": float(np.min(ray_residuals)),
                    "mean": float(np.mean(ray_residuals)),
                    "max": float(np.max(ray_residuals)),
                },
            }
            out_json = json_out_dir / "fused_head_v2.json"
            out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

            for angle, info in per_angle_top.items():
                qimg = Path(info["query_image"])
                if not qimg.exists():
                    continue
                pred = reproj.get(angle, {})
                t1 = info["top1_patch"]
                draw_overlay(
                    query_image=qimg,
                    out_path=viz_dir / f"{qimg.stem}_{angle}_fused_v2.png",
                    pred_nx=float(pred.get("pred_patch_center_x_norm", 0.5)),
                    pred_ny=float(pred.get("pred_patch_center_y_norm", 0.5)),
                    obs_nx=float(t1["patch_center_x_norm"]),
                    obs_ny=float(t1["patch_center_y_norm"]),
                )

            np.save(data_dir / "fused_head_xyz_voxel.npy", np.array(voxel_xyz, dtype=float))
            np.save(data_dir / "ray_residuals.npy", ray_residuals.astype(float))
            total += 1
            log(f"    Saved fused v2 result ({len(per_angle_top)} angles): {out_json}")

    log(f"Done. Fused individuals: {total}")


if __name__ == "__main__":
    main()
