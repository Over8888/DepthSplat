from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


@dataclass(frozen=True)
class PresetConfig:
    name: str
    display_name: str
    experiment: str
    dataset_root: Path
    checkpoint_path: Path
    fixed_index_path: Path
    num_context_views: int
    image_shape: tuple[int, int]
    ori_image_shape: tuple[int, int] | None = None
    extra_overrides: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    sample_stage: str = "test"
    sample_limit: int = 32
    expected_min_views: int = 2


@dataclass(frozen=True)
class Settings:
    app_root: Path
    outputs_root: Path
    depthsplat_root: Path
    depthsplat_python: str
    host: str
    port: int
    cors_allow_origins: list[str]
    default_preset: str
    cancellation_grace_seconds: float
    log_tail_lines: int
    presets: dict[str, PresetConfig]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    app_root = Path(os.getenv("DEPTHSPLAT_V3_ROOT", "/root/autodl-tmp/demo_v2"))
    outputs_root = app_root / "outputs" / "tasks"
    depthsplat_root = Path(os.getenv("DEPTHSPLAT_ROOT", "/root/depthsplat"))
    default_python = "/root/miniconda3/envs/depthsplat/bin/python"
    if not Path(default_python).exists():
        default_python = sys.executable

    re10k_2view = PresetConfig(
        name="re10k_2view_256x256",
        display_name="RE10K 2-view 256x256",
        experiment="re10k",
        dataset_root=depthsplat_root / "datasets" / "re10k_360p",
        checkpoint_path=depthsplat_root / "pretrained" / "depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth",
        fixed_index_path=depthsplat_root / "assets" / "evaluation_index_re10k_video.json",
        num_context_views=2,
        image_shape=(256, 256),
        extra_overrides=[
            "dataset.test_chunk_interval=100",
            "model.encoder.num_scales=2",
            "model.encoder.upsample_factor=2",
            "model.encoder.lowest_feature_resolution=4",
            "model.encoder.monodepth_vit_type=vitl",
            "data_loader.test.num_workers=0",
            "wandb.mode=disabled",
            "trainer.num_sanity_val_steps=0",
            "dataset.skip_bad_shape=false",
        ],
    )
    re10k_6view = PresetConfig(
        name="re10k_6view_512x960",
        display_name="RE10K 6-view 512x960",
        experiment="dl3dv",
        dataset_root=depthsplat_root / "datasets" / "re10k_720p_subset",
        checkpoint_path=depthsplat_root / "pretrained" / "depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth",
        fixed_index_path=depthsplat_root / "assets" / "re10k_ctx_6v_video.json",
        num_context_views=6,
        image_shape=(512, 960),
        ori_image_shape=(720, 1280),
        expected_min_views=6,
        extra_overrides=[
            "dataset.test_chunk_interval=1",
            "dataset.image_shape=[512,960]",
            "dataset.ori_image_shape=[720,1280]",
            "model.encoder.num_scales=2",
            "model.encoder.upsample_factor=4",
            "model.encoder.lowest_feature_resolution=8",
            "model.encoder.monodepth_vit_type=vitb",
            "model.encoder.gaussian_adapter.gaussian_scale_max=0.1",
            "data_loader.test.num_workers=0",
            "test.render_chunk_size=10",
            "wandb.mode=disabled",
            "trainer.num_sanity_val_steps=0",
            "dataset.skip_bad_shape=false",
        ],
    )
    dl3dv_6view = PresetConfig(
        name="dl3dv_6view_512x960",
        display_name="DL3DV 6-view 512x960",
        experiment="dl3dv",
        dataset_root=depthsplat_root / "datasets" / "dl3dv_960p_test_subset",
        checkpoint_path=depthsplat_root / "pretrained" / "depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth",
        fixed_index_path=depthsplat_root / "assets" / "dl3dv_start_0_distance_50_ctx_6v_video_0_50.json",
        num_context_views=6,
        image_shape=(512, 960),
        ori_image_shape=(540, 960),
        expected_min_views=6,
        env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        extra_overrides=[
            "dataset.test_chunk_interval=1",
            "dataset.image_shape=[512,960]",
            "dataset.ori_image_shape=[540,960]",
            "model.encoder.upsample_factor=8",
            "model.encoder.lowest_feature_resolution=8",
            "model.encoder.gaussian_adapter.gaussian_scale_max=0.1",
            "data_loader.test.num_workers=0",
            "test.stablize_camera=true",
            "test.render_chunk_size=1",
            "wandb.mode=disabled",
            "trainer.num_sanity_val_steps=0",
            "dataset.skip_bad_shape=false",
        ],
    )
    presets = {p.name: p for p in (re10k_2view, re10k_6view, dl3dv_6view)}
    default_preset = os.getenv("DEPTHSPLAT_V3_PRESET", re10k_6view.name)
    if default_preset not in presets:
        default_preset = re10k_6view.name
    return Settings(
        app_root=app_root,
        outputs_root=outputs_root,
        depthsplat_root=depthsplat_root,
        depthsplat_python=os.getenv("DEPTHSPLAT_PYTHON", default_python),
        host=os.getenv("DEPTHSPLAT_V3_HOST", "0.0.0.0"),
        port=int(os.getenv("DEPTHSPLAT_V3_PORT", "8012")),
        cors_allow_origins=_parse_csv_env(
            "DEPTHSPLAT_V3_CORS_ORIGINS",
            [
                "http://127.0.0.1:5174",
                "http://localhost:5174",
                "http://192.168.6.1:5174"
            ],
        ),
        default_preset=default_preset,
        cancellation_grace_seconds=float(os.getenv("DEPTHSPLAT_V3_CANCEL_GRACE", "5")),
        log_tail_lines=int(os.getenv("DEPTHSPLAT_V3_LOG_TAIL", "200")),
        presets=presets,
    )
