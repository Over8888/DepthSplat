from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import ToTensor

from app.config import Settings
from app.utils.logging import get_logger

_DIR = Path("/root/depthsplat")
_src_dir = _DIR / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from src.model.encoder.encoder_depthsplat import EncoderDepthSplat, EncoderDepthSplatCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.encoder.visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from src.dataset.shims.patch_shim import apply_patch_shim_to_views
from src.misc.image_io import save_image
from src.visualization.vis_depth import viz_depth_tensor
from src.model.ply_export import export_ply
from src.inference.pose_loader import estimate_near_far

MODEL_SIZES = {
    "small": {"monodepth_vit_type": "vits"},
    "base": {"monodepth_vit_type": "vits"},
    "large": {"monodepth_vit_type": "vitb"},
}


class DummyDatasetCfg:
    def __init__(self, background_color: list[float]):
        self.background_color = background_color
        self.image_shape = None


def _build_encoder_cfg(
    model_size: str = "base",
    num_depth_candidates: int = 128,
    sh_degree: int = 2,
    shim_patch_size: int = 4,
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
            gaussian_scale_max=3.0,
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
        num_scales=1,
        upsample_factor=4,
        lowest_feature_resolution=4,
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


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, prefix: str = "") -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    if prefix:
        plen = len(prefix)
        state_dict = {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        missing_filt = [k for k in missing if "visualizer" not in k]
        if missing_filt:
            print(f"Warning: missing keys ({len(missing_filt)}): {missing_filt[:5]}...")


class DemoInference:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self._encoder: EncoderDepthSplat | None = None
        self._decoder: DecoderSplattingCUDA | None = None
        self._device = torch.device("cuda")

    @property
    def encoder(self) -> EncoderDepthSplat:
        if self._encoder is None:
            raise RuntimeError("DemoInference model not loaded; call load_model() first")
        return self._encoder

    @property
    def decoder(self) -> DecoderSplattingCUDA:
        if self._decoder is None:
            raise RuntimeError("DemoInference model not loaded; call load_model() first")
        return self._decoder

    def load_model(self) -> None:
        if self._encoder is not None:
            return
        checkpoint_path = self.settings.demo_checkpoint_path
        self.logger.info("loading demo model", event="demo_model_loading", fields={"checkpoint": str(checkpoint_path)})

        encoder_cfg = _build_encoder_cfg(
            model_size=self.settings.demo_model_size,
            num_depth_candidates=self.settings.demo_num_depth_candidates,
            sh_degree=self.settings.demo_sh_degree,
            shim_patch_size=self.settings.demo_shim_patch_size,
        )
        encoder = EncoderDepthSplat(encoder_cfg)
        decoder_cfg = DecoderSplattingCUDACfg(name="splatting_cuda")
        dataset_cfg = DummyDatasetCfg(background_color=[0.0, 0.0, 0.0])
        decoder = DecoderSplattingCUDA(decoder_cfg, dataset_cfg)  # type: ignore[arg-type]

        _load_checkpoint(encoder, checkpoint_path, prefix="encoder.")
        _load_checkpoint(decoder, checkpoint_path, prefix="decoder.")

        encoder.to(self._device).eval()
        decoder.to(self._device).eval()

        self._encoder = encoder
        self._decoder = decoder
        self.logger.info("demo model loaded", event="demo_model_loaded")

    def run(
        self,
        task_dir: Path,
        output_dir: Path,
        scene_key: str,
        image_paths: list[str],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        context_indices: list[int],
        target_indices: list[int],
        options: dict[str, Any] | None = None,
    ) -> None:
        self.load_model()
        options = options or {}

        save_video = bool(options.get("saveVideo", True))
        save_depth = bool(options.get("exportDepthMap", True))
        save_ply = True

        output_dir.mkdir(parents=True, exist_ok=True)

        images: list[torch.Tensor] = [
            ToTensor()(Image.open(p).convert("RGB"))[:3] for p in image_paths
        ]
        V = len(images)

        context_extrinsics = extrinsics
        context_intrinsics = intrinsics

        context_near, context_far = estimate_near_far(context_extrinsics)

        context_image = torch.stack(images)
        context = {
            "extrinsics": context_extrinsics.to(self._device),
            "intrinsics": context_intrinsics.to(self._device),
            "image": context_image.to(self._device),
            "near": context_near.to(self._device),
            "far": context_far.to(self._device),
        }
        context = {k: v.unsqueeze(0) for k, v in context.items()}

        divisor = self.encoder.cfg.shim_patch_size * self.encoder.cfg.downscale_factor
        context = apply_patch_shim_to_views(context, divisor)
        _, _, _, h_new, w_new = context["image"].shape

        visualization_dump: dict = {}
        with torch.no_grad():
            result = self.encoder.forward(
                context,
                global_step=0,
                deterministic=False,
                visualization_dump=visualization_dump,
            )

        if isinstance(result, dict):
            pred_depths = result["depths"]
            gaussians = result["gaussians"]
        else:
            pred_depths = None
            gaussians = result

        depth_dir = output_dir / "images" / scene_key / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        if save_depth and pred_depths is not None:
            depth = pred_depths[0]
            for i, d in enumerate(depth):
                stem = Path(image_paths[i]).stem
                d_viz = viz_depth_tensor(1.0 / d.cpu(), return_numpy=True)
                Image.fromarray(d_viz).save(depth_dir / f"{i:06d}.png")

        do_render = save_video and target_indices
        rendered_dir: Path | None = None
        if do_render:
            rendered_dir = output_dir / "images" / scene_key / "renders"
            rendered_dir.mkdir(parents=True, exist_ok=True)

            tgt_list = list(target_indices)
            target_extrinsics = context_extrinsics[tgt_list]
            target_intrinsics = context_intrinsics[tgt_list]
            target_near = context_near[:1].expand(len(tgt_list))
            target_far = context_far[:1].expand(len(tgt_list))

            with torch.no_grad():
                rendered = self.decoder.forward(
                    gaussians,
                    target_extrinsics.unsqueeze(0).to(self._device),
                    target_intrinsics.unsqueeze(0).to(self._device),
                    target_near.unsqueeze(0).to(self._device),
                    target_far.unsqueeze(0).to(self._device),
                    (h_new, w_new),
                )
            colors = rendered.color[0]
            for i, c in enumerate(colors):
                save_image(c, rendered_dir / f"{tgt_list[i]:06d}.png")

        if save_ply and "visualization_dump" in visualization_dump:
            ply_dir = output_dir / "gaussians"
            ply_dir.mkdir(parents=True, exist_ok=True)

            v, _, h, w = context["image"].shape[1:]
            means = rearrange(gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w)
            mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
            GAUSSIAN_TRIM = 8
            mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

            def trim(element):
                element = rearrange(element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w)
                return element[mask][None]

            vis = visualization_dump
            cam_rotations = trim(vis["rotations"])[0]
            c2w_mat = repeat(
                context["extrinsics"][0, :, :3, :3],
                "v a b -> h w spp v a b",
                h=h, w=w, spp=1,
            )
            c2w_mat = c2w_mat[mask]

            cam_rotations_np = R.from_quat(cam_rotations.detach().cpu().numpy()).as_matrix()
            world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
            world_rotations = R.from_matrix(world_mat).as_quat()
            world_rotations = torch.from_numpy(world_rotations).to(vis["scales"])

            export_ply(
                context_extrinsics[0],
                trim(gaussians.means)[0],
                trim(vis["scales"])[0],
                world_rotations,
                trim(gaussians.harmonics)[0],
                trim(gaussians.opacities)[0],
                ply_dir / f"{scene_key}.ply",
            )

        if save_video and rendered_dir is not None:
            videos_dir = output_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)

            indices_str = "_".join(str(i) for i in target_indices)
            output_video = videos_dir / f"{scene_key}_frame_{indices_str}.mp4"
            img_pattern = rendered_dir / "%06d.png"
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", "24",
                "-i", str(img_pattern),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(output_video),
            ]
            result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0 and rendered_dir.exists():
                pass

        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        benchmark = {
            "encoder": [],
            "decoder": [],
        }
        (metrics_dir / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")

        self.logger.info(
            "demo inference completed",
            event="demo_inference_done",
            fields={"scene_key": scene_key, "num_views": V},
        )
