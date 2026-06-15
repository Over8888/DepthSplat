from __future__ import annotations

import gc
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import (
    OUTPUT_DIR,
    SCENE_KEYS,
    SCENE_CHUNK_FILES,
    CONTEXT_INDICES,
    DEPTHSPLAT_RES,
    MVSPLAT_RES,
    VGGT_RES,
    DA3_RES,
    VIEW_ANGLES,
    MAX_POINTS_PER_CLOUD,
    MAX_POINTS_PER_RENDER,
    DBSCAN_EPS_RATIO,
    DBSCAN_MIN_SAMPLES,
    DATASET_ROOT,
)
from .backproject import load_scene_images, load_gt_cameras, convert_poses, scale_intrinsics, back_project_multi_view
from .inference.depthsplat_pmr import run_depthsplat_pmr, load_depthsplat_depth
from .inference.mvsplat import run_mvsplat, load_mvsplat_depth
from .inference.vggt import run_vggt_inference
from .inference.da3 import run_da3_inference
from .anomaly import detect_floating_points, detect_surface_breaks
from .visualize import render_4x4_grid


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idxs = np.random.choice(len(points), max_points, replace=False)
    return points[idxs]


def run_experiment():
    print("=" * 60)
    print("Experiment V2: Multi-View Point Cloud Comparison")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    for scene_key in SCENE_KEYS:
        print(f"\n{'=' * 60}")
        print(f"Scene: {scene_key}")
        print(f"Context: {CONTEXT_INDICES[scene_key]}")
        print(f"{'=' * 60}")

        chunk_file = DATASET_ROOT / "test" / SCENE_CHUNK_FILES[scene_key]
        cidx = CONTEXT_INDICES[scene_key]

        # Load ground truth data
        images = load_scene_images(str(chunk_file), scene_key)
        gt_cameras_18d = load_gt_cameras(str(chunk_file), scene_key)
        c2w_gt, intrinsics_norm_gt = convert_poses(gt_cameras_18d)
        print(f"Loaded {len(images)} images, cameras: {gt_cameras_18d.shape}")

        context_images = [images[i] for i in cidx]
        context_c2w = c2w_gt[cidx].numpy()
        context_intrinsics_norm = intrinsics_norm_gt[cidx].numpy()

        gt_depth_near = 0.1
        gt_depth_far = 1000.0

        method_points = {}
        method_colors = {}
        method_anomalies = {}

        # ============================================================
        # Method 1: DepthSplat + PMR (Ours)
        # ============================================================
        print("\n[1/4] DepthSplat+PMR...")
        torch.cuda.empty_cache()
        gc.collect()

        ds_output_dir = OUTPUT_DIR / scene_key / "depthsplat_pmr"
        ds_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            depth_npy_dir = run_depthsplat_pmr(
                scene_key, cidx, DEPTHSPLAT_RES, ds_output_dir
            )
            depths_dict = load_depthsplat_depth(depth_npy_dir)

            target_resolutions = []
            depths_list = []
            for idx in cidx:
                if idx in depths_dict:
                    depths_list.append(depths_dict[idx])
                    target_resolutions.append(DEPTHSPLAT_RES)

            if depths_list:
                pts = back_project_multi_view(
                    depths_list,
                    context_intrinsics_norm,
                    context_c2w,
                    target_resolutions,
                    stride=1,
                    depth_threshold=gt_depth_far,
                )
                pts = _downsample(pts, MAX_POINTS_PER_CLOUD)
            else:
                print("[DepthSplat] No depth maps found!")
                pts = np.zeros((0, 3), dtype=np.float32)

            scene_scale = np.linalg.norm(pts.std(axis=0)) if len(pts) > 10 else 1.0
            eps = max(scene_scale * DBSCAN_EPS_RATIO, 0.05)

            float_anomaly = detect_floating_points(
                pts, eps=eps, min_samples=DBSCAN_MIN_SAMPLES
            )
            break_anomaly = detect_surface_breaks(pts)

            method_points["Ours (PMR)"] = pts
            method_colors["Ours (PMR)"] = None
            method_anomalies["Ours (PMR)"] = {
                **float_anomaly,
                **break_anomaly,
            }
            print(f"[DepthSplat] {len(pts)} points, scene_scale={scene_scale:.3f}, "
                  f"floating_clusters={len(float_anomaly['floating_centroids'])}")
        except Exception as e:
            print(f"[DepthSplat] FAILED: {e}")
            method_points["Ours (PMR)"] = np.zeros((0, 3), dtype=np.float32)
            method_colors["Ours (PMR)"] = None
            method_anomalies["Ours (PMR)"] = {}

        # ============================================================
        # Method 2: MVSplat
        # ============================================================
        print("\n[2/4] MVSplat...")
        torch.cuda.empty_cache()
        gc.collect()

        mv_output_dir = OUTPUT_DIR / scene_key / "mvsplat"
        mv_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            depth_npy_dir = run_mvsplat(
                scene_key, cidx, MVSPLAT_RES, mv_output_dir
            )
            depths_dict = load_mvsplat_depth(depth_npy_dir)

            target_resolutions = []
            depths_list = []
            for idx in cidx:
                if idx in depths_dict:
                    depths_list.append(depths_dict[idx])
                    target_resolutions.append(MVSPLAT_RES)

            if depths_list:
                pts = back_project_multi_view(
                    depths_list,
                    context_intrinsics_norm,
                    context_c2w,
                    target_resolutions,
                    stride=1,
                    depth_threshold=gt_depth_far,
                )
                pts = _downsample(pts, MAX_POINTS_PER_CLOUD)
            else:
                pts = np.zeros((0, 3), dtype=np.float32)

            scene_scale = np.linalg.norm(pts.std(axis=0)) if len(pts) > 10 else 1.0
            eps = max(scene_scale * DBSCAN_EPS_RATIO, 0.05)

            float_anomaly = detect_floating_points(
                pts, eps=eps, min_samples=DBSCAN_MIN_SAMPLES
            )
            break_anomaly = detect_surface_breaks(pts)

            method_points["MVSplat"] = pts
            method_colors["MVSplat"] = None
            method_anomalies["MVSplat"] = {**float_anomaly, **break_anomaly}
            print(f"[MVSplat] {len(pts)} points")
        except Exception as e:
            print(f"[MVSplat] FAILED: {e}")
            method_points["MVSplat"] = np.zeros((0, 3), dtype=np.float32)
            method_colors["MVSplat"] = None
            method_anomalies["MVSplat"] = {}

        # ============================================================
        # Method 3: VGGT
        # ============================================================
        print("\n[3/4] VGGT...")
        torch.cuda.empty_cache()
        gc.collect()

        try:
            vggt_result = run_vggt_inference(context_images, device="cuda")
            vggt_depth = vggt_result["depth"]             # (1, S, H, W) or (S, H, W)
            vggt_depth_conf = vggt_result["depth_conf"]
            vggt_h, vggt_w = vggt_result["image_shape"]

            if vggt_depth.ndim == 4:
                vggt_depth = vggt_depth[0]
                vggt_depth_conf = vggt_depth_conf[0]

            vggt_K = scale_intrinsics(
                torch.from_numpy(context_intrinsics_norm),
                vggt_h, vggt_w,
            ).numpy()

            pts_list = []
            for i in range(len(cidx)):
                depth_i = vggt_depth[i]
                conf_i = vggt_depth_conf[i]

                H, W = depth_i.shape
                K = vggt_K[i]
                c2w = context_c2w[i]
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]

                yy, xx = np.mgrid[0:H:2, 0:W:2]
                xx = xx.flatten()
                yy = yy.flatten()
                d = depth_i[yy, xx]
                conf_vals = conf_i[yy, xx]

                valid = (d > 1e-3) & np.isfinite(d) & (conf_vals > 0.1) & (d < gt_depth_far)
                xx, yy, d = xx[valid], yy[valid], d[valid]

                x_cam = (xx - cx) / fx * d
                y_cam = (yy - cy) / fy * d
                z_cam = d
                ones = np.ones_like(x_cam)
                cam_points = np.stack([x_cam, y_cam, z_cam, ones], axis=0)
                world_points_i = (c2w @ cam_points)[:3, :].T
                pts_list.append(world_points_i)

            if pts_list:
                pts = np.concatenate(pts_list, axis=0)
                pts = _downsample(pts, MAX_POINTS_PER_CLOUD)
            else:
                pts = np.zeros((0, 3), dtype=np.float32)

            scene_scale = np.linalg.norm(pts.std(axis=0)) if len(pts) > 10 else 1.0
            eps = max(scene_scale * DBSCAN_EPS_RATIO, 0.05)

            float_anomaly = detect_floating_points(
                pts, eps=eps, min_samples=DBSCAN_MIN_SAMPLES
            )
            break_anomaly = detect_surface_breaks(pts)

            method_points["VGGT"] = pts
            method_colors["VGGT"] = None
            method_anomalies["VGGT"] = {**float_anomaly, **break_anomaly}
            print(f"[VGGT] {len(pts)} points, image_shape=({vggt_h},{vggt_w})")
        except Exception as e:
            print(f"[VGGT] FAILED: {e}")
            method_points["VGGT"] = np.zeros((0, 3), dtype=np.float32)
            method_colors["VGGT"] = None
            method_anomalies["VGGT"] = {}

        # ============================================================
        # Method 4: DA3 (Depth Anything v3 BASE)
        # ============================================================
        print("\n[4/4] DA3 (BASE)...")
        torch.cuda.empty_cache()
        gc.collect()

        try:
            da3_images_uint8 = [(img * 255).astype(np.uint8) for img in context_images]
            da3_result = run_da3_inference(da3_images_uint8, device="cuda")

            da3_depth = da3_result["depth"]
            da3_conf = da3_result["conf"]
            da3_h, da3_w = da3_result["image_shape"]

            da3_K = scale_intrinsics(
                torch.from_numpy(context_intrinsics_norm),
                da3_h, da3_w,
            ).numpy()

            pts_list = []
            for i in range(len(cidx)):
                depth_i = da3_depth[i]
                conf_i = da3_conf[i] if da3_conf is not None else np.ones_like(depth_i)
                K = da3_K[i]
                c2w = context_c2w[i]

                H, W = depth_i.shape
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]

                yy, xx = np.mgrid[0:H:2, 0:W:2]
                xx = xx.flatten()
                yy = yy.flatten()
                d = depth_i[yy, xx]
                conf_vals = conf_i[yy, xx]

                valid = (d > 1e-3) & np.isfinite(d) & (conf_vals > 0.1) & (d < gt_depth_far)
                xx, yy, d = xx[valid], yy[valid], d[valid]

                x_cam = (xx - cx) / fx * d
                y_cam = (yy - cy) / fy * d
                z_cam = d
                ones = np.ones_like(x_cam)
                cam_points = np.stack([x_cam, y_cam, z_cam, ones], axis=0)
                world_points_i = (c2w @ cam_points)[:3, :].T
                pts_list.append(world_points_i)

            if pts_list:
                pts = np.concatenate(pts_list, axis=0)
                pts = _downsample(pts, MAX_POINTS_PER_CLOUD)
            else:
                pts = np.zeros((0, 3), dtype=np.float32)

            scene_scale = np.linalg.norm(pts.std(axis=0)) if len(pts) > 10 else 1.0
            eps = max(scene_scale * DBSCAN_EPS_RATIO, 0.05)

            float_anomaly = detect_floating_points(
                pts, eps=eps, min_samples=DBSCAN_MIN_SAMPLES
            )
            break_anomaly = detect_surface_breaks(pts)

            method_points["DA3"] = pts
            method_colors["DA3"] = None
            method_anomalies["DA3"] = {**float_anomaly, **break_anomaly}
            print(f"[DA3] {len(pts)} points")
        except Exception as e:
            print(f"[DA3] FAILED: {e}")
            import traceback
            traceback.print_exc()
            method_points["DA3"] = np.zeros((0, 3), dtype=np.float32)
            method_colors["DA3"] = None
            method_anomalies["DA3"] = {}

        # ============================================================
        # Save intermediate results
        # ============================================================
        save_dir = OUTPUT_DIR / scene_key
        np.savez_compressed(
            save_dir / "points.npz",
            **{k.replace(" ", "_"): v for k, v in method_points.items()},
        )

        # ============================================================
        # Render 4x4 visualization
        # ============================================================
        print("\nRendering 4x4 visualization...")
        render_4x4_grid(
            method_points=method_points,
            method_colors=method_colors,
            method_anomalies=method_anomalies,
            view_angles=VIEW_ANGLES,
            output_path=str(save_dir / f"{scene_key}_comparison.png"),
            max_points=MAX_POINTS_PER_RENDER,
            title=f"Point Cloud Comparison — {scene_key}",
        )

        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"Done! Results in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_experiment()
