#!/usr/bin/env python
"""DA3-BASE depth inference subprocess script.

Usage:
  python da3_infer.py --image-dir <dir> --output-dir <dir>
Input:  <image-dir>/image_*.png (2 images)
Output: <output-dir>/depth_*.npy, <output-dir>/cameras.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def w2c_to_c2w_3x4(w2c: np.ndarray) -> np.ndarray:
    b = w2c.shape[0]
    w2c_4x4 = np.tile(np.eye(4), (b, 1, 1))
    w2c_4x4[:, :3, :] = w2c[:, :3, :]
    c2w = np.linalg.inv(w2c_4x4)
    return c2w[:, :3, :4]


def pixel_to_normalized_intrinsics(K_px: np.ndarray, H: int, W: int) -> np.ndarray:
    K_norm = K_px.copy()
    K_norm[:, 0, 0] /= W
    K_norm[:, 0, 2] /= W
    K_norm[:, 1, 1] /= H
    K_norm[:, 1, 2] /= H
    return K_norm


def load_da3_model(device: torch.device):
    sys.path.insert(0, "/root/autodl-tmp/Depth-Anything-3/src")
    from depth_anything_3.api import DepthAnything3
    import safetensors.torch

    model = DepthAnything3(model_name="da3-base")
    ckpt_path = "/root/autodl-tmp/Depth-Anything-3/model.safetensors"
    state = safetensors.torch.load_file(ckpt_path, device="cpu")
    stripped = {}
    for k, v in state.items():
        if k.startswith("model."):
            stripped[k[6:]] = v
        else:
            stripped[k] = v
    model.model.load_state_dict(stripped, strict=False)
    print("DA3: loaded state_dict with model. prefix stripped")

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.image_dir.glob("image_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No image_*.png found in {args.image_dir}")

    device = torch.device("cuda")
    model = load_da3_model(device)

    from PIL import Image
    images_rgb = []
    images_uint8 = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images_uint8.append(np.array(img))
        images_rgb.append(np.array(img).astype(np.float32) / 255.0)
    images_rgb_np = np.stack(images_rgb, axis=0)
    images_uint8_np = np.stack(images_uint8, axis=0)

    H_orig, W_orig = images_rgb_np.shape[1], images_rgb_np.shape[2]

    prediction = model.inference(
        [Image.fromarray(img) for img in images_uint8_np],
        process_res=504,
        process_res_method="upper_bound_resize",
        use_ray_pose=False,
        ref_view_strategy="first",
    )

    depths = prediction.depth
    extrinsics_w2c = prediction.extrinsics
    intrinsics_px = prediction.intrinsics

    if depths is not None and hasattr(depths, 'ndim'):
        depth_np = np.asarray(depths)
    else:
        raise RuntimeError("DA3 produced no depth")

    predicted_H, predicted_W = depth_np.shape[1], depth_np.shape[2]

    c2w = w2c_to_c2w_3x4(extrinsics_w2c)
    K_norm = pixel_to_normalized_intrinsics(intrinsics_px, predicted_H, predicted_W)

    for i in range(len(depth_np)):
        np.save(args.output_dir / f"depth_{i:06d}.npy", depth_np[i])
        Image.fromarray(images_uint8_np[i]).save(args.output_dir / f"image_{i:06d}.png")

    np.savez(
        args.output_dir / "cameras.npz",
        extrinsics=c2w,
        intrinsics=K_norm,
        near=np.array(1.0),
        far=np.array(100.0),
        source="da3_predicted",
    )

    del model
    torch.cuda.empty_cache()
    print("DA3 inference done.")


if __name__ == "__main__":
    main()
