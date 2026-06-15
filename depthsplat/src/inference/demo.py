"""
Independent inference pipeline for DepthSplat.

Usage:
    python -m src.inference.demo \\
        --checkpoint pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth \\
        --images ./scene/ \\
        --poses ./scene/transforms.json \\
        --target_poses ./target_poses.json \\
        --output ./output/ \\
        --render_depth \\
        --save_ply \\
        --seed 42
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat

from .pose_loader import (
    estimate_near_far,
    extract_frame_id,
    load_camera_jsons_from_range,
    load_image_to_tensor,
    load_poses,
    load_single_camera_json,
    load_target_poses,
)

DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(DIR))

from src.model.encoder.encoder_depthsplat import (
    EncoderDepthSplat,
    EncoderDepthSplatCfg,
)
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.encoder.visualization.encoder_visualizer_depthsplat_cfg import (
    EncoderVisualizerDepthSplatCfg,
)
from src.model.decoder.decoder_splatting_cuda import (
    DecoderSplattingCUDA,
    DecoderSplattingCUDACfg,
)
from src.dataset.shims.patch_shim import apply_patch_shim_to_views
from src.misc.image_io import save_image, save_video
from src.visualization.vis_depth import viz_depth_tensor
from src.model.ply_export import export_ply
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from scipy.spatial.transform import Rotation
from PIL import Image


@dataclass
class DummyDatasetCfg:
    """Minimal dataset config to satisfy DecoderSplattingCUDA requirements."""

    background_color: list[float]
    image_shape: list[int] | None = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

MODEL_SIZES = {
    "small": {"monodepth_vit_type": "vits"},
    "base": {"monodepth_vit_type": "vitb"},
    "large": {"monodepth_vit_type": "vitl"},
}


def build_encoder_cfg(
    model_size: str = "base",
    num_depth_candidates: int = 128,
    sh_degree: int = 2,
    shim_patch_size: int = 4,
    gaussian_scale_max: float = 3.0,
    upsample_factor: int = 2,
    lowest_feature_resolution: int = 4,
    num_scales: int = 2,
) -> EncoderDepthSplatCfg:
    ms = MODEL_SIZES.get(model_size, MODEL_SIZES["base"])
    return EncoderDepthSplatCfg(
        name="depthsplat",
        d_feature=128,
        num_depth_candidates=num_depth_candidates,
        num_surfaces=1,
        visualizer=EncoderVisualizerDepthSplatCfg(
            num_samples=8,
            min_resolution=256,
            export_ply=False,
        ),
        gaussian_adapter=GaussianAdapterCfg(
            gaussian_scale_min=1e-10,
            gaussian_scale_max=gaussian_scale_max,
            sh_degree=sh_degree,
        ),
        gaussians_per_pixel=1,
        unimatch_weights_path=None,
        downscale_factor=4,
        shim_patch_size=shim_patch_size,
        multiview_trans_attn_split=2,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=[1, 1, 1],
        costvolume_unet_attn_res=[4],
        depth_unet_feat_dim=32,
        depth_unet_attn_res=[16],
        depth_unet_channel_mult=[1, 1, 1, 1, 1],
        num_scales=num_scales,
        upsample_factor=upsample_factor,
        lowest_feature_resolution=lowest_feature_resolution,
        depth_unet_channels=128,
        grid_sample_disable_cudnn=False,
        large_gaussian_head=False,
        color_large_unet=False,
        init_sh_input_img=True,
        feature_upsampler_channels=64,
        gaussian_regressor_channels=64,
        supervise_intermediate_depth=False,
        return_depth=True,
        train_depth_only=False,
        cost_volume_confidence=False,
        pmr_guided_smooth=False,
        monodepth_vit_type=ms["monodepth_vit_type"],
        local_mv_match=2,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, prefix: str = "") -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    if prefix:
        plen = len(prefix)
        state_dict = {
            k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys ({len(missing)}): {[k for k in missing[:5]]}...")
    if unexpected:
        print(f"Warning: unexpected keys ({len(unexpected)}): {[k for k in unexpected[:5]]}...")


def build_model(
    checkpoint_path: Path,
    model_size: str = "base",
    num_depth_candidates: int = 128,
    sh_degree: int = 2,
    shim_patch_size: int = 4,
    gaussian_scale_max: float = 3.0,
    upsample_factor: int = 2,
    lowest_feature_resolution: int = 4,
    num_scales: int = 2,
    background_color: tuple = (0.0, 0.0, 0.0),
    device: torch.device = torch.device("cuda"),
) -> tuple[EncoderDepthSplat, DecoderSplattingCUDA]:
    encoder_cfg = build_encoder_cfg(
        model_size=model_size,
        num_depth_candidates=num_depth_candidates,
        sh_degree=sh_degree,
        shim_patch_size=shim_patch_size,
        gaussian_scale_max=gaussian_scale_max,
        upsample_factor=upsample_factor,
        lowest_feature_resolution=lowest_feature_resolution,
        num_scales=num_scales,
    )
    encoder = EncoderDepthSplat(encoder_cfg)
    decoder_cfg = DecoderSplattingCUDACfg(name="splatting_cuda")
    dataset_cfg = DummyDatasetCfg(background_color=list(background_color))
    decoder = DecoderSplattingCUDA(decoder_cfg, dataset_cfg)  # type: ignore[arg-type]

    load_checkpoint(encoder, checkpoint_path, prefix="encoder.")
    load_checkpoint(decoder, checkpoint_path, prefix="decoder.")

    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(
    encoder: EncoderDepthSplat,
    decoder: DecoderSplattingCUDA,
    images: list[torch.Tensor],  # each [3, H, W] in [0, 1]
    context_extrinsics: torch.Tensor,  # [V, 4, 4] c2w OpenCV
    context_intrinsics: torch.Tensor,  # [V, 3, 3] normalized
    context_near: Optional[torch.Tensor] = None,  # [V]
    context_far: Optional[torch.Tensor] = None,  # [V]
    target_extrinsics: Optional[torch.Tensor] = None,  # [V_tgt, 4, 4]
    target_intrinsics: Optional[torch.Tensor] = None,  # [V_tgt, 3, 3]
    target_near: Optional[torch.Tensor] = None,
    target_far: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    V = len(images)
    assert context_extrinsics.shape[0] == V
    assert context_intrinsics.shape[0] == V

    # Compute near/far if not provided
    if context_near is None or context_far is None:
        near, far = estimate_near_far(context_extrinsics)
        context_near = near if context_near is None else context_near
        context_far = far if context_far is None else context_far

    # Build context dict: batch=1, view=V
    context_image = torch.stack(images)  # [V, 3, H, W]
    context = {
        "extrinsics": context_extrinsics.to(device),  # [V, 4, 4]
        "intrinsics": context_intrinsics.to(device),  # [V, 3, 3]
        "image": context_image.to(device),            # [V, 3, H, W]
        "near": context_near.to(device),
        "far": context_far.to(device),
    }
    context = {k: v.unsqueeze(0) for k, v in context.items()}  # [1, V, ...]

    # Apply patch_shim: crop to dimensions divisible by patch_size * downscale_factor = 16
    divisor = encoder.cfg.shim_patch_size * encoder.cfg.downscale_factor  # 16
    context = apply_patch_shim_to_views(context, divisor)
    _, _, _, h_new, w_new = context["image"].shape

    visualization_dump: dict = {}
    with torch.no_grad():
        result = encoder.forward(
            context,
            global_step=0,
            deterministic=False,
            visualization_dump=visualization_dump,
        )

    if isinstance(result, dict):
        pred_depths = result["depths"]  # [1, V, H, W]
        gaussians = result["gaussians"]
    else:
        pred_depths = None
        gaussians = result

    outputs: dict = {
        "gaussians": gaussians,
        "pred_depths": pred_depths,
        "visualization_dump": visualization_dump,
        "context": context,
    }

    # Render target views if requested
    if target_extrinsics is not None:
        assert target_intrinsics is not None
        V_tgt = target_extrinsics.shape[0]
        if target_near is None:
            target_near = context_near[:1].expand(V_tgt)
        if target_far is None:
            target_far = context_far[:1].expand(V_tgt)

        with torch.no_grad():
            rendered = decoder.forward(
                gaussians,
                target_extrinsics.unsqueeze(0).to(device),
                target_intrinsics.unsqueeze(0).to(device),
                target_near.unsqueeze(0).to(device),
                target_far.unsqueeze(0).to(device),
                (h_new, w_new),
            )
        outputs["rendered_color"] = rendered.color  # [1, V_tgt, 3, H, W]

        depth_rendered = decoder.forward(
            gaussians,
            target_extrinsics.unsqueeze(0).to(device),
            target_intrinsics.unsqueeze(0).to(device),
            target_near.unsqueeze(0).to(device),
            target_far.unsqueeze(0).to(device),
            (h_new, w_new),
            depth_mode="depth",
        )
        outputs["rendered_depth"] = depth_rendered.depth  # [1, V_tgt, H, W]

    return outputs


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_outputs(
    outputs: dict,
    output_dir: Path,
    image_paths: list[Path],
    context_extrinsics: torch.Tensor,
    context_indices: list[int] | None = None,
    target_indices: list[int] | None = None,
    scene: str = "scene",
    save_depth: bool = True,
    save_ply: bool = True,
    save_renders: bool = True,
    save_depth_npy: bool = False,
    save_depth_concat: bool = False,
    save_video_flag: bool = False,
    video_fps: int = 30,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_depth and outputs["pred_depths"] is not None:
        depth = outputs["pred_depths"][0].detach().cpu()
        context_image = outputs["context"]["image"][0].detach().cpu()

        depth_dir = output_dir / "images" / scene / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)

        if save_depth_concat:
            image = rearrange(context_image, "b c h w -> h (b w) c")
            image_concat = (image.numpy() * 255).astype(np.uint8)
            depth_concat_list = []

        for i, d in enumerate(depth):
            idx = context_indices[i] if context_indices else i
            d_viz = viz_depth_tensor(1.0 / d, return_numpy=True)
            Image.fromarray(d_viz).save(depth_dir / f"{idx:06d}.png")
            if save_depth_npy:
                np.save(depth_dir / f"{idx:06d}.npy", d.numpy())
            if save_depth_concat:
                depth_concat_list.append(d_viz)

        if save_depth_concat and depth_concat_list:
            depth_concat = np.concatenate(depth_concat_list, axis=1)
            concat = np.concatenate((image_concat, depth_concat), axis=0)
            Image.fromarray(concat).save(depth_dir / f"img_depth_{scene}.png")

    if save_renders and "rendered_color" in outputs:
        color_dir = output_dir / "images" / scene / "color"
        color_dir.mkdir(parents=True, exist_ok=True)
        colors = outputs["rendered_color"][0]
        for i, c in enumerate(colors):
            idx = target_indices[i] if target_indices else i
            save_image(c, color_dir / f"{idx:06d}.png")

    if save_video_flag and "rendered_color" in outputs:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        ctx_ids = "_".join(str(x) for x in context_indices) if context_indices else "ctx"
        colors = outputs["rendered_color"][0]
        save_video(
            [c for c in colors],
            video_dir / f"{scene}_frame_{ctx_ids}.mp4",
            fps=video_fps,
        )

    if save_ply and "visualization_dump" in outputs:
        gaussians = outputs["gaussians"]
        vis = outputs["visualization_dump"]
        ply_dir = output_dir / "gaussians"
        ply_dir.mkdir(parents=True, exist_ok=True)

        v, _, h, w = outputs["context"]["image"].shape[1:]

        means = rearrange(
            gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w
        )
        mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
        GAUSSIAN_TRIM = 8
        mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

        def trim(element):
            element = rearrange(element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w)
            return element[mask][None]

        cam_rotations = trim(vis["rotations"])[0]
        c2w_mat = repeat(
            outputs["context"]["extrinsics"][0, :, :3, :3],
            "v a b -> h w spp v a b",
            h=h,
            w=w,
            spp=1,
        )
        c2w_mat = c2w_mat[mask]

        cam_rotations_np = Rotation.from_quat(cam_rotations.detach().cpu().numpy()).as_matrix()
        world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
        world_rotations = Rotation.from_matrix(world_mat).as_quat()
        world_rotations = torch.from_numpy(world_rotations).to(vis["scales"])

        export_ply(
            context_extrinsics[0].to(gaussians.means.device),
            trim(gaussians.means)[0],
            trim(vis["scales"])[0],
            world_rotations,
            trim(gaussians.harmonics)[0],
            trim(gaussians.opacities)[0],
            ply_dir / f"{scene}.ply",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DepthSplat inference on custom images + poses")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained .pth checkpoint path")
    parser.add_argument("--images", nargs="+", type=Path, help="Image file paths or a directory to glob")
    parser.add_argument("--poses", type=Path, default=None, help="Pose file (transforms.json, .npz, etc.)")
    parser.add_argument("--pose-format", type=str, default=None,
                        choices=["transforms_json", "colmap_images_txt", "npz", "npy"],
                        help="Force pose format (by default auto-detected)")
    parser.add_argument("--colmap-cameras", type=Path, default=None, help="COLMAP cameras.txt")
    parser.add_argument("--colmap-images", type=Path, default=None, help="COLMAP images.txt")
    parser.add_argument("--target-poses", type=Path, default=None,
                        help="Pose file for novel view rendering targets")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--model-size", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--num-depth-candidates", type=int, default=128)
    parser.add_argument("--sh-degree", type=int, default=2)
    parser.add_argument("--shim-patch-size", type=int, default=4,
                        help="Shim patch size (4 for re10k, 16 for dl3dv)")
    parser.add_argument("--gaussian-scale-max", type=float, default=3.0,
                        help="Max Gaussian scale (3.0 for re10k, 0.1 for re10kdl3dv)")
    parser.add_argument("--upsample-factor", type=int, default=2,
                        help="CNN backbone upsample factor (2 for re10k, 4 for re10kdl3dv)")
    parser.add_argument("--num-scales", type=int, default=2,
                        help="Number of feature scales (1 for small re10k, 2 for base/large re10k)")
    parser.add_argument("--lowest-feature-resolution", type=int, default=4,
                        help="Lowest CNN feature resolution (4 for re10k, 8 for re10kdl3dv)")
    parser.add_argument("--background-color", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--render-depth", action="store_true", default=True,
                        help="Save depth maps for input views")
    parser.add_argument("--save-ply", action="store_true", default=False,
                        help="Export 3D Gaussians as PLY file")
    parser.add_argument("--save-renders", action="store_true", default=True,
                        help="Save rendered target views")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--max-image-size", type=int, default=256,
                        help="Resize images so max(H,W) <= this value (default 256)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory with all frames + camera.json files (enables continuous sequence mode)")
    parser.add_argument("--context-images", nargs="+", type=Path, default=None,
                        help="Context image paths (for continuous sequence mode)")
    parser.add_argument("--context-cameras", nargs="+", type=Path, default=None,
                        help="Camera.json paths; auto-matched by _camera.json suffix if omitted")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene name (auto-detected from data-dir if not set)")
    parser.add_argument("--save-depth-npy", action="store_true", default=False,
                        help="Save raw depth as .npy files")
    parser.add_argument("--save-depth-concat", action="store_true", default=False,
                        help="Save concatenated depth visualization")
    parser.add_argument("--save-video", action="store_true", default=False,
                        help="Save rendered target views as MP4 video")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="Video frame rate (default 30)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Continuous sequence mode ----
    if args.data_dir is not None and args.context_images is not None:
        cont_images = args.context_images
        if args.context_cameras is not None:
            cont_cameras = args.context_cameras
            assert len(cont_images) == len(cont_cameras), \
                f"Number of context images ({len(cont_images)}) != cameras ({len(cont_cameras)})"
        else:
            cont_cameras = [p.parent / (p.stem + "_camera.json") for p in cont_images]

        images = [load_image_to_tensor(p) for p in cont_images]

        ctx_ext, ctx_int, frame_ids = [], [], []
        for cpath in cont_cameras:
            cdata = load_single_camera_json(cpath)
            ctx_ext.append(cdata["extrinsics"])
            ctx_int.append(cdata["intrinsics"])
            frame_ids.append(extract_frame_id(cpath.name))

        context_extrinsics = torch.stack(ctx_ext)
        context_intrinsics = torch.stack(ctx_int)
        min_id, max_id = min(frame_ids), max(frame_ids)

        target_pose_data = load_camera_jsons_from_range(args.data_dir, min_id, max_id)
        target_extrinsics = target_pose_data.extrinsics
        target_intrinsics = target_pose_data.intrinsics
        target_near, target_far = estimate_near_far(target_extrinsics)

        context_near, context_far = estimate_near_far(context_extrinsics)
        if args.near is not None:
            context_near = torch.full_like(context_near, args.near)
        if args.far is not None:
            context_far = torch.full_like(context_far, args.far)

        image_paths_for_save = cont_images
        context_indices = frame_ids
        target_indices = list(range(min_id, max_id + 1))
        scene_name = args.scene or args.data_dir.name
        print(f"Continuous mode: {len(images)} context views, "
              f"{target_extrinsics.shape[0]} target views (frames {min_id}-{max_id})")

    # ---- Original mode ----
    else:
        image_paths_arg = args.images or []
        if len(image_paths_arg) == 1 and image_paths_arg[0].is_dir():
            images_dir = image_paths_arg[0]
            image_paths = None
        else:
            images_dir = None
            image_paths = image_paths_arg if image_paths_arg else None

        pose_data = load_poses(
            pose_path=args.poses,
            images_dir=images_dir,
            colmap_cameras=args.colmap_cameras,
            colmap_images=args.colmap_images,
            format=args.pose_format,
            image_paths=image_paths,
        )

        images = [load_image_to_tensor(p) for p in pose_data.image_paths]

        target_extrinsics = None
        target_intrinsics = None
        target_near = None
        target_far = None
        if args.target_poses is not None:
            target_pose_data = load_target_poses(args.target_poses, format=args.pose_format)
            target_extrinsics = target_pose_data.extrinsics
            target_intrinsics = target_pose_data.intrinsics
            target_near, target_far = estimate_near_far(target_extrinsics)

        context_extrinsics = pose_data.extrinsics
        context_intrinsics = pose_data.intrinsics

        context_near, context_far = estimate_near_far(context_extrinsics)
        if args.near is not None:
            context_near = torch.full_like(context_near, args.near)
        if args.far is not None:
            context_far = torch.full_like(context_far, args.far)

        image_paths_for_save = pose_data.image_paths
        context_indices = None
        target_indices = None
        scene_name = args.scene or "scene"

    # Resize images to fit GPU memory (normalized intrinsics are resolution-invariant)
    from torchvision.transforms import functional as TF
    for i in range(len(images)):
        _, h, w = images[i].shape
        scale = args.max_image_size / max(h, w)
        if scale < 1.0:
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            images[i] = TF.resize(images[i], [new_h, new_w],
                                  interpolation=TF.InterpolationMode.BILINEAR)

    # ---- Shared: Build model, run inference, save ----
    print(f"Loading checkpoint from {args.checkpoint}...")
    encoder, decoder = build_model(
        checkpoint_path=args.checkpoint,
        model_size=args.model_size,
        num_depth_candidates=args.num_depth_candidates,
        sh_degree=args.sh_degree,
        shim_patch_size=args.shim_patch_size,
        gaussian_scale_max=args.gaussian_scale_max,
        upsample_factor=args.upsample_factor,
        lowest_feature_resolution=args.lowest_feature_resolution,
        num_scales=args.num_scales,
        background_color=tuple(args.background_color),
        device=device,
    )

    print(f"Running inference on {len(images)} views...")
    outputs = run_inference(
        encoder=encoder,
        decoder=decoder,
        images=images,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        context_near=context_near,
        context_far=context_far,
        target_extrinsics=target_extrinsics,
        target_intrinsics=target_intrinsics,
        target_near=target_near,
        target_far=target_far,
        device=device,
        seed=args.seed,
    )

    print(f"Saving outputs to {args.output}...")
    save_outputs(
        outputs=outputs,
        output_dir=args.output,
        image_paths=image_paths_for_save,
        context_extrinsics=context_extrinsics,
        context_indices=context_indices,
        target_indices=target_indices,
        scene=scene_name,
        save_depth=args.render_depth,
        save_ply=args.save_ply,
        save_renders=args.save_renders,
        save_depth_npy=args.save_depth_npy,
        save_depth_concat=args.save_depth_concat,
        save_video_flag=args.save_video,
        video_fps=args.video_fps,
    )

    # Compute metrics if ground truth is available (continuous sequence mode)
    if args.data_dir is not None and args.context_images is not None \
            and target_extrinsics is not None and "rendered_color" in outputs:
        from torchvision.transforms import functional as TF

        rendered = outputs["rendered_color"][0]  # [V_tgt, 3, H, W]
        _, _, rh, rw = rendered.shape

        gt_list = []
        for i in range(target_extrinsics.shape[0]):
            fidx = min_id + i
            fname = f"{fidx:05d}.jpg"
            gt_path = args.data_dir / fname
            if gt_path.exists():
                gt_img = load_image_to_tensor(gt_path)[:3]
                _, gh, gw = gt_img.shape
                if (gh, gw) != (rh, rw):
                    gt_img = TF.resize(gt_img, [rh, rw],
                                       interpolation=TF.InterpolationMode.BILINEAR)
                gt_list.append(gt_img)

        if gt_list:
            gt = torch.stack(gt_list).to(device)
            pred = rendered.to(device)
            psnr_val = compute_psnr(gt, pred).mean().item()
            ssim_val = compute_ssim(gt, pred).mean().item()
            lpips_val = compute_lpips(gt, pred).mean().item()
            metrics = {
                "psnr": round(psnr_val, 4),
                "ssim": round(ssim_val, 4),
                "lpips": round(lpips_val, 4),
                "num_context_views": len(images),
                "num_target_views": target_extrinsics.shape[0],
                "frame_range": [min_id, max_id],
            }
            with (args.output / "metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
        else:
            print("Warning: No ground truth images found, skipping metrics")

    print("Done.")


if __name__ == "__main__":
    main()
