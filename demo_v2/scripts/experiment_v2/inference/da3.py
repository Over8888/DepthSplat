from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..config import DA3_ROOT, DA3_PRESET


def _ensure_da3_importable():
    src_path = str(DA3_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _setup_da3_config():
    config_path = DA3_ROOT / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps({"model_name": DA3_PRESET}))


def run_da3_inference(
    images: list[np.ndarray],
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    _setup_da3_config()

    import sys
    import types

    _fake_packages = [
        "pycolmap",
        "pycolmap.absolute_pose_estimation",
        "pycolmap.homography_decomposition",
        "pycolmap.two_view_geometry",
        "pycolmap.sfm",
        "pycolmap.camera",
        "evo",
        "evo.core",
        "evo.core.trajectory",
        "depth_anything_3.utils.pose_align",
        "depth_anything_3.utils.export",
        "depth_anything_3.utils.export.gs",
        "depth_anything_3.utils.export.gs_video",
        "depth_anything_3.utils.export.colmap",
        "depth_anything_3.utils.export.glb",
        "depth_anything_3.utils.camera_trj_helpers",
        "depth_anything_3.utils.camera_trj_helpers_legacy",
    ]
    for mod_name in _fake_packages:
        if mod_name not in sys.modules:
            fake_mod = types.ModuleType(mod_name)
            fake_mod.__path__ = []
            sys.modules[mod_name] = fake_mod

    exp_mod = sys.modules["depth_anything_3.utils.export"]
    exp_mod.export = lambda *a, **kw: None
    exp_mod.export_to_gs_ply = lambda *a, **kw: None
    exp_mod.export_to_gs_video = lambda *a, **kw: None
    exp_mod.export_to_colmap = lambda *a, **kw: None
    exp_mod.export_to_glb = lambda *a, **kw: None

    pa_mod = sys.modules["depth_anything_3.utils.pose_align"]
    pa_mod.align_poses_umeyama = lambda *a, **kw: None

    _ensure_da3_importable()

    import torch
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(str(DA3_ROOT)).to(device)
    model.eval()

    with torch.inference_mode():
        prediction = model.inference(image=images)

    depth = prediction.depth.astype(np.float32)
    conf = prediction.conf.astype(np.float32) if prediction.conf is not None else np.ones_like(depth)
    extrinsics_w2c = prediction.extrinsics.astype(np.float32)
    intrinsics = prediction.intrinsics.astype(np.float32)
    H_out, W_out = depth.shape[-2:]

    del model, prediction
    torch.cuda.empty_cache()

    return {
        "depth": depth,
        "conf": conf,
        "extrinsics_w2c": extrinsics_w2c,
        "intrinsics": intrinsics,
        "image_shape": (H_out, W_out),
    }
