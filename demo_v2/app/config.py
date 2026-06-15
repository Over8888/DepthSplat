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


def _parse_optional_path_env(name: str) -> Path | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return Path(value)


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
    source_image_shape: tuple[int, int] | None = None
    extra_overrides: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    sample_stage: str = "test"
    sample_limit: int = 10
    expected_min_views: int = 2
    max_auto_video_frames: int | None = None


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
    vggt_root: Path
    vggt_checkpoint_path: Path | None
    vggt_model_id: str
    vggt_load_resolution: int
    vggt_inference_resolution: int
    vggt_device: str | None
    camera_backend: str
    ttt3r_root: Path
    ttt3r_python: str
    ttt3r_model_path: Path
    ttt3r_size: int
    ttt3r_model_update_type: str
    ttt3r_reset_interval: int
    ttt3r_device: str
    task_id_timezone: str
    demo_checkpoint_path: Path
    demo_model_size: str
    demo_shim_patch_size: int
    demo_num_depth_candidates: int
    demo_sh_degree: int
    presets: dict[str, PresetConfig]
    script_mapping: dict[tuple[str, int, str], str] = field(default_factory=dict)


def resolve_script_path(
    settings: "Settings",
    sample_id: str | None,
    preset_id: str | None,
    input_view_count: int | None,
) -> str | None:
    """
    Resolve script path based on input mode, view count, and quality.
    
    Args:
        settings: Settings instance with script_mapping
        sample_id: Sample ID (if present, use sample mode)
        preset_id: Preset ID (extract quality from it)
        input_view_count: Number of input views (2/4/6)
    
    Returns:
        Relative script path or None if not found
    """
    input_mode = "sample" if sample_id else "manual"
    
    if input_view_count is None:
        if preset_id and preset_id in settings.presets:
            input_view_count = settings.presets[preset_id].num_context_views
        else:
            input_view_count = 6
    
    quality = "base"
    if preset_id:
        preset_lower = preset_id.lower()
        if "large" in preset_lower:
            quality = "large"
        elif "small" in preset_lower:
            quality = "small"
        elif preset_id in settings.presets:
            ckpt_name = settings.presets[preset_id].checkpoint_path.name.lower()
            if "large" in ckpt_name:
                quality = "large"
            elif "small" in ckpt_name:
                quality = "small"
    
    key = (input_mode, input_view_count, quality)
    return settings.script_mapping.get(key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    app_root = Path(os.getenv("DEPTHSPLAT_V3_ROOT", "/root/autodl-tmp/demo_v2"))
    outputs_root = app_root / "outputs" / "tasks"
    depthsplat_root = Path(os.getenv("DEPTHSPLAT_ROOT", "/root/depthsplat"))
    default_python = "/root/miniconda3/envs/depthsplat/bin/python"
    if not Path(default_python).exists():
        default_python = sys.executable

    def make_preset(
        name: str,
        display_name: str,
        experiment: str,
        dataset_root: Path,
        checkpoint: str,
        index: str,
        num_context_views: int,
        image_shape: tuple[int, int],
        source_image_shape: tuple[int, int],
        ori_image_shape: tuple[int, int] | None = None,
        expected_min_views: int | None = None,
        max_auto_video_frames: int | None = None,
    ) -> PresetConfig:
        return PresetConfig(
            name=name,
            display_name=display_name,
            experiment=experiment,
            dataset_root=dataset_root,
            checkpoint_path=depthsplat_root / "pretrained" / checkpoint,
            fixed_index_path=depthsplat_root / "assets" / index,
            num_context_views=num_context_views,
            image_shape=image_shape,
            ori_image_shape=ori_image_shape,
            source_image_shape=source_image_shape,
            expected_min_views=expected_min_views or num_context_views,
            max_auto_video_frames=max_auto_video_frames,
        )

    re10k_360p_root = Path(os.getenv("DEPTHSPLAT_RE10K_2VIEW_ROOT", "/root/autodl-tmp/RealEstate10K"))
    re10k_720p_root = depthsplat_root / "datasets" / "re10k_720p"
    dl3dv_960p_root = Path(os.getenv("DEPTHSPLAT_DL3DV_ROOT", "/root/autodl-tmp/dl3dv_960p"))

    presets_list = [
        make_preset(
            "re10k_large_2view",
            "RE10K large 2-view 256x256",
            "re10k",
            re10k_360p_root,
            "depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth",
            "evaluation_index_re10k_video.json",
            2,
            (256, 256),
            (360, 640),
            max_auto_video_frames=256,
        ),
        make_preset(
            "re10k_base_2view",
            "RE10K base 2-view 256x256",
            "re10k",
            re10k_360p_root,
            "depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth",
            "evaluation_index_re10k_video.json",
            2,
            (256, 256),
            (360, 640),
            max_auto_video_frames=256,
        ),
        make_preset(
            "re10k_small_2view",
            "RE10K small 2-view 256x256",
            "re10k",
            re10k_360p_root,
            "depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth",
            "evaluation_index_re10k_video.json",
            2,
            (256, 256),
            (360, 640),
            max_auto_video_frames=256,
        ),
        make_preset(
            "re10k_large_4view",
            "RE10K large 4-view 512x960",
            "dl3dv",
            re10k_720p_root,
            "depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth",
            "re10k_ctx_4v_video.json",
            4,
            (512, 960),
            (720, 1280),
            ori_image_shape=(720, 1280),
        ),
        make_preset(
            "re10k_base_4view",
            "DL3DV base 4-view 512x960",
            "dl3dv",
            dl3dv_960p_root,
            "depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth",
            "dl3dv_start_0_distance_50_ctx_4v_video_0_50.json",
            4,
            (512, 960),
            (540, 960),
            ori_image_shape=(540, 960),
        ),
        make_preset(
            "re10k_small_4view",
            "RE10K small 4-view 512x960",
            "dl3dv",
            re10k_720p_root,
            "depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth",
            "re10k_ctx_4v_video.json",
            4,
            (512, 960),
            (720, 1280),
            ori_image_shape=(720, 1280),
        ),
        make_preset(
            "re10k_large_6view",
            "RE10K large 6-view 512x960",
            "dl3dv",
            re10k_720p_root,
            "depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth",
            "re10k_ctx_6v_video.json",
            6,
            (512, 960),
            (720, 1280),
            ori_image_shape=(720, 1280),
            max_auto_video_frames=279,
        ),
        make_preset(
            "re10k_base_6view",
            "DL3DV base 6-view 512x960",
            "dl3dv",
            dl3dv_960p_root,
            "depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth",
            "dl3dv_start_0_distance_50_ctx_6v_video_0_50.json",
            6,
            (512, 960),
            (540, 960),
            ori_image_shape=(540, 960),
            max_auto_video_frames=50,
        ),
        make_preset(
            "re10k_small_6view",
            "RE10K small 6-view 512x960",
            "dl3dv",
            re10k_720p_root,
            "depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth",
            "re10k_ctx_6v_video.json",
            6,
            (512, 960),
            (720, 1280),
            ori_image_shape=(720, 1280),
            max_auto_video_frames=279,
        ),
    ]
    presets = {p.name: p for p in presets_list}
    default_preset = os.getenv("DEPTHSPLAT_V3_PRESET", "re10k_base_6view")
    if default_preset not in presets:
        default_preset = "re10k_base_6view"
    
    # Script mapping: (input_mode, num_views, quality) -> script_path
    # input_mode: "sample" (RE10K dataset) or "manual" (user-provided poses)
    # num_views: 2, 4, 6
    # quality: "large", "base", "small"
    script_mapping = {
        # Sample mode (RE10K dataset evaluation)
        ("sample", 2, "large"): "scripts/best_official_configs/re10k_large_2view.sh",
        ("sample", 2, "base"): "scripts/best_official_configs/re10k_base_2view.sh",
        ("sample", 2, "small"): "scripts/best_official_configs/re10k_small_2view.sh",
        ("sample", 4, "large"): "scripts/best_official_configs/re10k_large_4view.sh",
        ("sample", 4, "base"): "scripts/best_official_configs/re10k_base_4view.sh",
        ("sample", 4, "small"): "scripts/best_official_configs/re10k_small_4view.sh",
        ("sample", 6, "large"): "scripts/best_official_configs/re10k_large_6view.sh",
        ("sample", 6, "base"): "scripts/best_official_configs/re10k_base_6view.sh",
        ("sample", 6, "small"): "scripts/best_official_configs/re10k_small_6view.sh",
        # Manual mode (user-provided images + poses)
        ("manual", 2, "large"): "scripts/best_official_configs/manual_base.sh",
        ("manual", 2, "base"): "scripts/best_official_configs/manual_base.sh",
        ("manual", 2, "small"): "scripts/best_official_configs/manual_base.sh",
        ("manual", 4, "large"): "scripts/best_official_configs/manual_base.sh",
        ("manual", 4, "base"): "scripts/best_official_configs/manual_base.sh",
        ("manual", 4, "small"): "scripts/best_official_configs/manual_base.sh",
        ("manual", 6, "large"): "scripts/best_official_configs/manual_base.sh",
        ("manual", 6, "base"): "scripts/best_official_configs/manual_base.sh",
        ("manual", 6, "small"): "scripts/best_official_configs/manual_base.sh",
    }
    
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
                "http://192.168.6.1:5174",
            ],
        ),
        default_preset=default_preset,
        cancellation_grace_seconds=float(os.getenv("DEPTHSPLAT_V3_CANCEL_GRACE", "5")),
        log_tail_lines=int(os.getenv("DEPTHSPLAT_V3_LOG_TAIL", "200")),
        vggt_root=Path(os.getenv("DEPTHSPLAT_VGGT_ROOT", "/root/autodl-tmp/vggt")),
        vggt_checkpoint_path=(
            _parse_optional_path_env("DEPTHSPLAT_VGGT_CHECKPOINT")
            or (Path("/root/autodl-tmp/model.pt") if Path("/root/autodl-tmp/model.pt").exists() else None)
        ),
        vggt_model_id=os.getenv("DEPTHSPLAT_VGGT_MODEL_ID", "facebook/VGGT-1B"),
        vggt_load_resolution=int(os.getenv("DEPTHSPLAT_VGGT_LOAD_RESOLUTION", "1024")),
        vggt_inference_resolution=int(os.getenv("DEPTHSPLAT_VGGT_INFERENCE_RESOLUTION", "518")),
        vggt_device=os.getenv("DEPTHSPLAT_VGGT_DEVICE") or None,
        camera_backend=os.getenv("DEPTHSPLAT_CAMERA_BACKEND", "ttt3r"),
        ttt3r_root=Path(os.getenv("DEPTHSPLAT_TTT3R_ROOT", "/root/TTT3R")),
        ttt3r_python=os.getenv("DEPTHSPLAT_TTT3R_PYTHON", os.getenv("DEPTHSPLAT_PYTHON", default_python)),
        ttt3r_model_path=Path(os.getenv("DEPTHSPLAT_TTT3R_MODEL_PATH", "/root/TTT3R/src/cut3r_512_dpt_4_64.pth")),
        ttt3r_size=int(os.getenv("DEPTHSPLAT_TTT3R_SIZE", "512")),
        ttt3r_model_update_type=os.getenv("DEPTHSPLAT_TTT3R_MODEL_UPDATE_TYPE", "ttt3r"),
        ttt3r_reset_interval=int(os.getenv("DEPTHSPLAT_TTT3R_RESET_INTERVAL", "200")),
        ttt3r_device=os.getenv("DEPTHSPLAT_TTT3R_DEVICE", "cuda"),
        task_id_timezone=os.getenv("DEPTHSPLAT_V3_TASK_TIMEZONE", "Asia/Shanghai"),
        demo_checkpoint_path=Path(os.getenv("DEPTHSPLAT_DEMO_CHECKPOINT", str(depthsplat_root / "pretrained" / "depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth"))),
        demo_model_size=os.getenv("DEPTHSPLAT_DEMO_MODEL_SIZE", "base"),
        demo_shim_patch_size=int(os.getenv("DEPTHSPLAT_DEMO_SHIM_PATCH_SIZE", "4")),
        demo_num_depth_candidates=int(os.getenv("DEPTHSPLAT_DEMO_NUM_DEPTH_CANDIDATES", "128")),
        demo_sh_degree=int(os.getenv("DEPTHSPLAT_DEMO_SH_DEGREE", "2")),
        presets=presets,
        script_mapping=script_mapping,
    )
