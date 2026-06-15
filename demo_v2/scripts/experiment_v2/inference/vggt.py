from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from ..config import VGGT_ROOT


VGGT_LOCAL_CHECKPOINT = Path("/root/autodl-tmp/model.pt")


def _ensure_vggt_importable():
    root = str(VGGT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def run_vggt_inference(
    images: list[np.ndarray],
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Run VGGT inference on a list of images.

    Args:
        images: list of (H, W, 3) numpy arrays in [0, 1]
        device: "cuda" or "cpu"

    Returns:
        dict with keys:
            depth: (S, H, W) metric depth maps
            depth_conf: (S, H, W) depth confidence
            world_points: (S, H, W, 3) world-space 3D points
            world_points_conf: (S, H, W)
            extrinsics_w2c: (S, 3, 4) world-to-camera extrinsics
            intrinsics: (S, 3, 3) camera intrinsics in pixel units
    """
    import tempfile
    from PIL import Image as PILImage

    _ensure_vggt_importable()

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    tmp_dir = tempfile.mkdtemp(prefix="vggt_imgs_")
    image_paths = []
    for i, img in enumerate(images):
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        path = f"{tmp_dir}/img_{i:04d}.png"
        PILImage.fromarray(img_uint8).save(path)
        image_paths.append(path)

    model = VGGT()
    state_dict = torch.load(str(VGGT_LOCAL_CHECKPOINT), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    processed = load_and_preprocess_images(image_paths).to(device)

    dtype = torch.bfloat16 if torch.cuda.is_available() and device == "cuda" else torch.float32
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(processed)

    S, _, H, W = processed.shape

    depth = predictions["depth"].squeeze(-1).float().cpu().numpy()
    depth_conf = predictions["depth_conf"].float().cpu().numpy()
    world_points = predictions["world_points"].float().cpu().numpy()
    world_points_conf = predictions["world_points_conf"].float().cpu().numpy()

    extrinsics_w2c, intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], (H, W)
    )
    extrinsics_w2c = extrinsics_w2c.float().cpu().numpy()
    intrinsics = intrinsics.float().cpu().numpy()

    del model, processed, predictions
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    torch.cuda.empty_cache()

    return {
        "depth": depth,
        "depth_conf": depth_conf,
        "world_points": world_points,
        "world_points_conf": world_points_conf,
        "extrinsics_w2c": extrinsics_w2c,
        "intrinsics": intrinsics,
        "image_shape": (H, W),
    }
