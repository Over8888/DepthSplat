#!/usr/bin/env python
"""VGGT depth inference subprocess script.

Usage:
  python vggt_infer.py --image-dir <dir> --output-dir <dir>
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


def w2c_3x4_to_c2w_4x4(w2c_3x4: np.ndarray) -> np.ndarray:
    b = w2c_3x4.shape[0]
    w2c_4x4 = np.tile(np.eye(4), (b, 1, 1))
    w2c_4x4[:, :3, :] = w2c_3x4
    c2w = np.linalg.inv(w2c_4x4)
    return c2w


def pixel_to_normalized_intrinsics(K_px: np.ndarray, H: int, W: int) -> np.ndarray:
    K_norm = K_px.copy()
    K_norm[:, 0, 0] /= W
    K_norm[:, 0, 2] /= W
    K_norm[:, 1, 1] /= H
    K_norm[:, 1, 2] /= H
    return K_norm


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, "/root/autodl-tmp/vggt")
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_paths = sorted(args.image_dir.glob("image_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No image_*.png found in {args.image_dir}")

    device = torch.device("cuda")

    model = VGGT()
    state = torch.load("/root/autodl-tmp/model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    images = load_and_preprocess_images(image_paths, mode="pad")
    images = images.to(device)
    H, W = images.shape[-2:]

    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            aggregated_tokens_list, ps_idx = model.aggregator(images[None])
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            depth, depth_conf = model.depth_head(
                aggregated_tokens_list, images[None], ps_idx,
                frames_chunk_size=2,
            )

    extrinsics_w2c, intrinsics_px = pose_encoding_to_extri_intri(pose_enc, (H, W))

    extrinsics_w2c_np = extrinsics_w2c[0].cpu().float().numpy()
    intrinsics_px_np = intrinsics_px[0].cpu().float().numpy()
    depth_np = depth[0, :, :, :, 0].cpu().float().numpy()

    c2w = w2c_3x4_to_c2w_4x4(extrinsics_w2c_np)[:, :3, :4]
    K_norm = pixel_to_normalized_intrinsics(intrinsics_px_np, H, W)

    images_vis = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    for i in range(len(depth_np)):
        np.save(args.output_dir / f"depth_{i:06d}.npy", depth_np[i])

    np.savez(
        args.output_dir / "cameras.npz",
        extrinsics=c2w,
        intrinsics=K_norm,
        near=np.array(1.0),
        far=np.array(100.0),
        source="vggt_predicted",
    )

    del model
    torch.cuda.empty_cache()
    print("VGGT inference done.")


if __name__ == "__main__":
    main()
