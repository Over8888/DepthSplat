from __future__ import annotations

import gc
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize

from .config import (
    EXP4_ROOT,
    SCENE_KEYS,
    SCENE_CHUNK_FILES,
    CONTEXT_INDICES,
    DATASET_ROOT,
    DEPTHSPLAT_RES,
    MVSPLAT_RES,
    VGGT_RES,
    DA3_RES,
    OPACITY_FILTER_THRESHOLD_GS,
    OPACITY_FILTER_THRESHOLD_PC,
    MAX_POINTS_PER_CLOUD,
)
from .backproject import (
    load_scene_images,
    load_gt_cameras,
    convert_poses,
    scale_intrinsics,
    back_project_depth,
)
from .inference.depthsplat_pmr import run_depthsplat_pmr, load_depthsplat_depth
from .inference.mvsplat import run_mvsplat, load_mvsplat_depth
from .inference.vggt import run_vggt_inference
from .inference.da3 import run_da3_inference
from .ply_export import (
    read_ply,
    write_gaussian_ply,
    write_pointcloud_ply,
    compute_max_eigenvalue,
)
from .gaussian_diagnostics import GaussianData, render_diagnostic_grid


def _downsample_points(pts: np.ndarray, max_points: int) -> np.ndarray:
    if len(pts) <= max_points:
        return pts
    idxs = np.random.choice(len(pts), max_points, replace=False)
    return pts[idxs]


def _rgb_from_sh_dc(sh_dc: np.ndarray) -> np.ndarray:
    sh_0 = 0.28209479177387814
    colors = sh_dc * sh_0 + 0.5
    return np.clip(colors, 0, 1)


def _sample_colors(centers_3d: np.ndarray, pixel_xy: np.ndarray,
                   image: np.ndarray) -> np.ndarray:
    H, W = image.shape[:2]
    colors = np.zeros((len(centers_3d), 3), dtype=np.float32)
    for i, (px, py) in enumerate(pixel_xy):
        pyi = int(np.clip(py, 0, H - 1))
        pxi = int(np.clip(px, 0, W - 1))
        colors[i] = image[pyi, pxi]
    return colors


def _resize_depth_and_conf(depth: np.ndarray, conf: np.ndarray,
                           intrinsics: np.ndarray,
                           orig_H: int, orig_W: int,
                           target_H: int, target_W: int):
    K = intrinsics.copy()
    K[0, 0] *= target_W / orig_W
    K[0, 2] *= target_W / orig_W
    K[1, 1] *= target_H / orig_H
    K[1, 2] *= target_H / orig_H
    depth_rs = resize(depth.astype(np.float32), (target_H, target_W), order=1)
    conf_rs = resize(conf.astype(np.float32), (target_H, target_W), order=1)
    return depth_rs, conf_rs, K


def _backproject_with_colors(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    image: np.ndarray,
    stride: int = 2,
    depth_threshold: float = 1000.0,
    max_points: int = 80000,
    conf_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    yy, xx = np.mgrid[0:H, 0:W]
    xx_f = xx.flatten().astype(np.float32)
    yy_f = yy.flatten().astype(np.float32)
    d = depth.flatten()

    valid = (d > 1e-3) & np.isfinite(d) & (d < depth_threshold)
    xx_f, yy_f, d = xx_f[valid], yy_f[valid], d[valid]

    if len(xx_f) > max_points:
        idxs = np.random.choice(len(xx_f), max_points, replace=False)
        xx_f, yy_f, d = xx_f[idxs], yy_f[idxs], d[idxs]

    x_cam = (xx_f - cx) / fx * d
    y_cam = (yy_f - cy) / fy * d
    z_cam = d
    ones = np.ones_like(x_cam)
    cam_points = np.stack([x_cam, y_cam, z_cam, ones], axis=0)
    world_points = (c2w @ cam_points)[:3, :].T.astype(np.float32)

    pixel_xy = np.stack([xx_f, yy_f], axis=1).astype(np.float32)
    colors = _sample_colors(world_points, pixel_xy, image)

    confidences = None
    if conf_map is not None:
        H_c, W_c = conf_map.shape
        px_int = np.clip(xx_f.astype(np.int32), 0, W_c - 1)
        py_int = np.clip(yy_f.astype(np.int32), 0, H_c - 1)
        confidences = conf_map[py_int, px_int].astype(np.float32)

    return world_points, colors, confidences


def run_experiment4():
    print("=" * 60)
    print("Experiment 4: Gaussian / Point Cloud Diagnostics")
    print(f"Output: {EXP4_ROOT}")
    print("=" * 60)

    method_order = ["Ours (PMR)", "MVSplat", "VGGT", "DA3"]

    for scene_key in SCENE_KEYS:
        print(f"\n{'=' * 60}")
        print(f"Scene: {scene_key}")
        print(f"Context: {CONTEXT_INDICES[scene_key]}")
        print(f"{'=' * 60}")

        chunk_file = DATASET_ROOT / "test" / SCENE_CHUNK_FILES[scene_key]
        cidx = CONTEXT_INDICES[scene_key]

        images = load_scene_images(str(chunk_file), scene_key)
        gt_cameras_18d = load_gt_cameras(str(chunk_file), scene_key)
        c2w_gt, intrinsics_norm_gt = convert_poses(gt_cameras_18d)
        print(f"Loaded {len(images)} images")

        scene_dir = EXP4_ROOT / scene_key
        scene_dir.mkdir(parents=True, exist_ok=True)

        all_data: dict[str, GaussianData] = {}
        intrinsics_per_method: dict[str, np.ndarray] = {}
        w2c_per_method: dict[str, np.ndarray] = {}
        H_per_method: dict[str, int] = {}
        W_per_method: dict[str, int] = {}

        view_idx = cidx[0]

        # ---- DepthSplat + PMR ----
        print("\n[1/4] DepthSplat+PMR...")
        torch.cuda.empty_cache()
        gc.collect()

        ds_dir = scene_dir / "depthsplat_pmr"
        ds_dir.mkdir(parents=True, exist_ok=True)

        try:
            depth_npy_dir = run_depthsplat_pmr(scene_key, cidx, DEPTHSPLAT_RES, ds_dir)
            depths_dict = load_depthsplat_depth(depth_npy_dir)

            ply_path = ds_dir / "gaussians" / f"{scene_key}.ply"
            if ply_path.exists():
                gs_data = read_ply(str(ply_path))
                centers = np.stack([gs_data["x"], gs_data["y"], gs_data["z"]], axis=1)

                opacities_raw = gs_data["opacity"]
                opacities = 1.0 / (1.0 + np.exp(-opacities_raw.astype(np.float64)))
                opacities = np.clip(opacities, 0.0, 1.0).astype(np.float32)

                scales = np.stack([
                    np.exp(gs_data["scale_0"].astype(np.float64)),
                    np.exp(gs_data["scale_1"].astype(np.float64)),
                    np.exp(gs_data["scale_2"].astype(np.float64)),
                ], axis=1).astype(np.float32)

                rotations = np.stack([gs_data["rot_0"], gs_data["rot_1"], gs_data["rot_2"], gs_data["rot_3"]], axis=1)
                sh_dc = np.stack([gs_data["f_dc_0"], gs_data["f_dc_1"], gs_data["f_dc_2"]], axis=1)
                rgb_dc = _rgb_from_sh_dc(sh_dc)
                max_eigenvalues = compute_max_eigenvalue(scales)

                centers = _downsample_points(centers, MAX_POINTS_PER_CLOUD)
                opacities = opacities[:len(centers)]
                scales = scales[:len(centers)]
                rotations = rotations[:len(centers)]
                sh_dc = sh_dc[:len(centers)]
                rgb_dc = rgb_dc[:len(centers)]
                max_eigenvalues = max_eigenvalues[:len(centers)]

                write_gaussian_ply(str(ds_dir / "full.ply"), centers, opacities, scales, rotations, rgb_dc)
                write_gaussian_ply(str(ds_dir / "filtered.ply"), centers, opacities, scales, rotations, rgb_dc,
                                   opacity_threshold=OPACITY_FILTER_THRESHOLD_GS)

                intrinsics_ctx_px = scale_intrinsics(
                    intrinsics_norm_gt[cidx[0]:cidx[0]+1], DEPTHSPLAT_RES[0], DEPTHSPLAT_RES[1]
                )[0].numpy()

                c2w_ctx = c2w_gt[cidx[0]].numpy()
                depth_img = depths_dict.get(cidx[0])
                if depth_img is not None:
                    img_ctx = images[cidx[0]]
                    if img_ctx.shape[:2] != DEPTHSPLAT_RES:
                        img_ctx = resize(img_ctx.astype(np.float32), DEPTHSPLAT_RES, order=1)
                    dcenters, dcolors, _ = _backproject_with_colors(depth_img, intrinsics_ctx_px, c2w_ctx, img_ctx)
                    dcenters = _downsample_points(dcenters, MAX_POINTS_PER_CLOUD)
                    dcolors = dcolors[:len(dcenters)]
                else:
                    dcenters = None
                    dcolors = None

                all_data["Ours (PMR)"] = GaussianData(
                    centers=centers, opacities=opacities, scales=scales,
                    max_eigenvalues=max_eigenvalues, sh_dc=rgb_dc,
                    has_real_gaussians=True, depth_map=None, conf_map=None,
                    method_name="Ours (PMR)",
                    display_centers=dcenters, display_sh_dc=dcolors,
                )
                intrinsics_per_method["Ours (PMR)"] = intrinsics_ctx_px
                w2c_per_method["Ours (PMR)"] = np.eye(4, dtype=np.float32)
                H_per_method["Ours (PMR)"] = DEPTHSPLAT_RES[0]
                W_per_method["Ours (PMR)"] = DEPTHSPLAT_RES[1]
                print(f"[DepthSplat] {len(centers)} gaussians loaded from PLY")
            else:
                print("[DepthSplat] WARNING: No PLY file found, backprojecting depth as pseudo-gaussians")
                raise FileNotFoundError("PLY not found - fall through to pseudo path")

        except Exception as e:
            print(f"[DepthSplat] PLY approach failed ({e}), falling back to pseudo-gaussians")
            try:
                depths_list = []
                for idx in cidx:
                    if idx in depths_dict:
                        depths_list.append(depths_dict[idx])

                intrinsics_ctx_px = scale_intrinsics(
                    intrinsics_norm_gt[cidx[0]:cidx[0]+1], DEPTHSPLAT_RES[0], DEPTHSPLAT_RES[1]
                )[0].numpy()
                c2w_ctx = c2w_gt[cidx[0]].numpy()
                w2c_ctx = np.linalg.inv(c2w_ctx)

                img_ctx = images[cidx[0]]
                if img_ctx.shape[:2] != DEPTHSPLAT_RES:
                    img_ctx = resize(img_ctx.astype(np.float32), DEPTHSPLAT_RES, order=1)

                depth_img = depths_dict[cidx[0]]
                centers, colors, _ = _backproject_with_colors(depth_img, intrinsics_ctx_px, c2w_ctx, img_ctx)
                centers = _downsample_points(centers, MAX_POINTS_PER_CLOUD)
                opacities = np.ones(len(centers), dtype=np.float32) * 0.8

                write_pointcloud_ply(str(ds_dir / "full.ply"), centers, colors, opacities)
                write_pointcloud_ply(str(ds_dir / "filtered.ply"), centers, colors, opacities,
                                     confidence_threshold=OPACITY_FILTER_THRESHOLD_PC)

                all_data["Ours (PMR)"] = GaussianData(
                    centers=centers, opacities=opacities, scales=None,
                    max_eigenvalues=None, sh_dc=colors,
                    has_real_gaussians=False, depth_map=depth_img, conf_map=None,
                    method_name="Ours (PMR)",
                )
                intrinsics_per_method["Ours (PMR)"] = intrinsics_ctx_px
                w2c_per_method["Ours (PMR)"] = w2c_ctx
                H_per_method["Ours (PMR)"] = DEPTHSPLAT_RES[0]
                W_per_method["Ours (PMR)"] = DEPTHSPLAT_RES[1]
                print(f"[DepthSplat] {len(centers)} pseudo-gaussians from depth")
            except Exception as e2:
                print(f"[DepthSplat] FAILED: {e2}")
                all_data["Ours (PMR)"] = None

        # ---- MVSplat ----
        print("\n[2/4] MVSplat...")
        torch.cuda.empty_cache()
        gc.collect()

        mv_dir = scene_dir / "mvsplat"
        mv_dir.mkdir(parents=True, exist_ok=True)

        try:
            depth_npy_dir = run_mvsplat(scene_key, cidx, MVSPLAT_RES, mv_dir)
            depths_dict = load_mvsplat_depth(depth_npy_dir)

            intrinsics_ctx_px = scale_intrinsics(
                intrinsics_norm_gt[cidx[0]:cidx[0]+1], MVSPLAT_RES[0], MVSPLAT_RES[1]
            )[0].numpy()
            c2w_ctx = c2w_gt[cidx[0]].numpy()
            w2c_ctx = np.linalg.inv(c2w_ctx)

            img_ctx = images[cidx[0]]
            if img_ctx.shape[:2] != MVSPLAT_RES:
                img_ctx = resize(img_ctx.astype(np.float32), MVSPLAT_RES, order=1)

            depth_img = depths_dict[cidx[0]]
            centers, colors, _ = _backproject_with_colors(depth_img, intrinsics_ctx_px, c2w_ctx, img_ctx)
            centers = _downsample_points(centers, MAX_POINTS_PER_CLOUD)
            opacities = np.ones(len(centers), dtype=np.float32) * 0.8

            write_pointcloud_ply(str(mv_dir / "full.ply"), centers, colors, opacities)
            write_pointcloud_ply(str(mv_dir / "filtered.ply"), centers, colors, opacities,
                                 confidence_threshold=OPACITY_FILTER_THRESHOLD_PC)

            all_data["MVSplat"] = GaussianData(
                centers=centers, opacities=opacities, scales=None,
                max_eigenvalues=None, sh_dc=colors,
                has_real_gaussians=False, depth_map=depth_img, conf_map=None,
                method_name="MVSplat",
            )
            intrinsics_per_method["MVSplat"] = intrinsics_ctx_px
            w2c_per_method["MVSplat"] = w2c_ctx
            H_per_method["MVSplat"] = MVSPLAT_RES[0]
            W_per_method["MVSplat"] = MVSPLAT_RES[1]
            print(f"[MVSplat] {len(centers)} pseudo-gaussians from depth")
        except Exception as e:
            print(f"[MVSplat] FAILED: {e}")
            all_data["MVSplat"] = None

        # ---- VGGT ----
        print("\n[3/4] VGGT...")
        torch.cuda.empty_cache()
        gc.collect()

        vggt_dir = scene_dir / "vggt"
        vggt_dir.mkdir(parents=True, exist_ok=True)

        try:
            context_images = [images[i] for i in cidx]
            vggt_result = run_vggt_inference(context_images, device="cuda")

            world_points = vggt_result["world_points"]
            world_points_conf = vggt_result["world_points_conf"]
            vggt_H, vggt_W = vggt_result["image_shape"]
            intrinsics_vggt = vggt_result["intrinsics"]
            extrinsics_w2c = vggt_result["extrinsics_w2c"]
            depth_vggt = vggt_result["depth"]

            vi = 0
            wp_flat = world_points[0, vi].reshape(-1, 3)
            conf_flat = world_points_conf[0, vi].reshape(-1)
            total_pixels = vggt_H * vggt_W
            pixel_indices = np.arange(total_pixels)

            valid = (conf_flat > 1e-3) & np.isfinite(wp_flat).all(axis=1)
            wp_flat = wp_flat[valid]
            conf_flat = conf_flat[valid]
            pixel_indices = pixel_indices[valid]

            if len(wp_flat) > MAX_POINTS_PER_CLOUD:
                idxs = np.random.choice(len(wp_flat), MAX_POINTS_PER_CLOUD, replace=False)
                wp = wp_flat[idxs]
                conf = conf_flat[idxs]
                pixel_indices = pixel_indices[idxs]
            else:
                wp = wp_flat
                conf = conf_flat

            img_resized = resize(context_images[vi].astype(np.float32), (vggt_H, vggt_W), order=1)
            colors = np.zeros((len(wp), 3), dtype=np.float32)
            for j, idx in enumerate(pixel_indices):
                py = int(idx) // vggt_W
                px = int(idx) % vggt_W
                colors[j] = img_resized[py, px]

            write_pointcloud_ply(str(vggt_dir / "full.ply"), wp, colors, conf)
            write_pointcloud_ply(str(vggt_dir / "filtered.ply"), wp, colors, conf,
                                 confidence_threshold=OPACITY_FILTER_THRESHOLD_PC)

            depth_img = depth_vggt[0, vi] if depth_vggt is not None else None

            c2w_vggt = np.linalg.inv(
                np.concatenate([
                    extrinsics_w2c[0, vi],
                    np.array([[0, 0, 0, 1]], dtype=np.float32),
                ], axis=0),
            )
            w2c_vggt = extrinsics_w2c[0, vi]

            all_data["VGGT"] = GaussianData(
                centers=wp, opacities=conf, scales=None,
                max_eigenvalues=None, sh_dc=colors,
                has_real_gaussians=False, depth_map=depth_img, conf_map=world_points_conf[0, vi],
                method_name="VGGT",
            )
            intrinsics_per_method["VGGT"] = intrinsics_vggt[0, vi]
            w2c_per_method["VGGT"] = w2c_vggt
            H_per_method["VGGT"] = vggt_H
            W_per_method["VGGT"] = vggt_W
            print(f"[VGGT] {len(wp)} points from world_points")
        except Exception as e:
            print(f"[VGGT] FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_data["VGGT"] = None

        # ---- DA3 ----
        print("\n[4/4] DA3...")
        torch.cuda.empty_cache()
        gc.collect()

        da3_dir = scene_dir / "da3"
        da3_dir.mkdir(parents=True, exist_ok=True)

        try:
            da3_images_uint8 = [(img * 255).astype(np.uint8) for img in context_images]
            da3_result = run_da3_inference(da3_images_uint8, device="cuda")

            da3_depth = da3_result["depth"]
            da3_conf = da3_result["conf"]
            da3_intrinsics = da3_result["intrinsics"]
            da3_extrinsics_w2c = da3_result["extrinsics_w2c"]
            da3_H, da3_W = da3_result["image_shape"]

            da3_c2w = np.linalg.inv(
                np.concatenate([
                    da3_extrinsics_w2c,
                    np.tile(np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 1, 4),
                            (da3_extrinsics_w2c.shape[0], 1, 1)),
                ], axis=1),
            )

            vi = 0
            K_da3 = da3_intrinsics[vi]
            c2w_da3 = da3_c2w[vi]
            w2c_da3 = da3_extrinsics_w2c[vi]
            depth_i = da3_depth[vi]
            conf_i = da3_conf[vi]

            img_resized = resize(context_images[vi].astype(np.float32), (da3_H, da3_W), order=1)

            centers, colors, conf_vals = _backproject_with_colors(
                depth_i, K_da3, c2w_da3, img_resized, conf_map=conf_i)
            centers = _downsample_points(centers, MAX_POINTS_PER_CLOUD)
            colors = colors[:len(centers)]
            conf_vals = conf_vals[:len(centers)]

            write_pointcloud_ply(str(da3_dir / "full.ply"), centers, colors, conf_vals)
            write_pointcloud_ply(str(da3_dir / "filtered.ply"), centers, colors, conf_vals,
                                 confidence_threshold=OPACITY_FILTER_THRESHOLD_PC)

            all_data["DA3"] = GaussianData(
                centers=centers, opacities=conf_vals, scales=None,
                max_eigenvalues=None, sh_dc=colors,
                has_real_gaussians=False, depth_map=depth_i, conf_map=conf_i,
                method_name="DA3",
            )
            intrinsics_per_method["DA3"] = K_da3
            w2c_per_method["DA3"] = w2c_da3
            H_per_method["DA3"] = da3_H
            W_per_method["DA3"] = da3_W
            print(f"[DA3] {len(centers)} pseudo-gaussians from depth")
        except Exception as e:
            print(f"[DA3] FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_data["DA3"] = None

        # ---- Render Diagnostics ----
        valid_data = {k: v for k, v in all_data.items() if v is not None}
        if len(valid_data) < 2:
            print(f"\n[Skipping] Not enough valid methods for {scene_key}")
            continue

        print("\nRendering diagnostic grid...")
        output_path = str(scene_dir / f"{scene_key}_diagnostics.png")
        render_diagnostic_grid(
            all_data=valid_data,
            intrinsics_per_method=intrinsics_per_method,
            w2c_per_method=w2c_per_method,
            H_per_method=H_per_method,
            W_per_method=W_per_method,
            view_idx=view_idx,
            output_path=output_path,
        )

        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"Done! Results in: {EXP4_ROOT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_experiment4()
