#!/usr/bin/env python3
"""
Run TTT3R on all scenes in image_sets and save per-image camera JSON files.

Each JSON maps to a specific image file with the naming pattern:
  <image_stem>_camera.json  (e.g. 00001_camera.json for 00001.jpg)

JSON format (OpenCV convention, compatible with depthsplat):
{
    "image_name": "00001.jpg",
    "width": 1920,
    "height": 1080,
    "fx": 1200.0,
    "fy": 1200.0,
    "cx": 960.0,
    "cy": 540.0,
    "camera_to_world": [[... 4x4 ...]]
}
"""

import os
import sys
import json
import glob
import time
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_ckpt_path import add_path_to_dust3r


def run_ttt3r_on_scene(scene_dir, model, device, size=512, reset_interval=1000000):
    """Run TTT3R inference on a single scene directory, save per-image JSON."""
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.utils.image import load_images
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import matrix_cumprod

    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    all_paths = sorted(glob.glob(os.path.join(scene_dir, '*')))
    img_paths = [p for p in all_paths if os.path.splitext(p.lower())[1] in img_extensions]

    if not img_paths:
        print(f"  No images found in {scene_dir}")
        return

    print(f"  {len(img_paths)} images")

    first_img = cv2.imread(img_paths[0])
    orig_h, orig_w = first_img.shape[:2]
    print(f"  Original resolution: {orig_w}x{orig_h}")

    images = load_images(img_paths, size=size, verbose=False)
    model_H = int(images[0]["true_shape"][0][0])
    model_W = int(images[0]["true_shape"][0][1])
    print(f"  Model resolution: {model_W}x{model_H}")

    views = []
    for i in range(len(images)):
        view = {
            "img": images[i]["img"],
            "ray_map": torch.full(
                (images[i]["img"].shape[0], 6,
                 images[i]["img"].shape[-2], images[i]["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
        }
        views.append(view)
        if (i + 1) % reset_interval == 0:
            from copy import deepcopy
            overlap_view = deepcopy(view)
            overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
            views.append(overlap_view)

    t0 = time.time()
    outputs, _ = inference_recurrent_lighter(views, model, device)
    elapsed = time.time() - t0
    print(f"  Inference: {elapsed:.1f}s ({elapsed/len(img_paths):.2f}s/frame)")

    outputs["pred"] = outputs["pred"][-len(img_paths):]
    outputs["views"] = outputs["views"][-len(img_paths):]

    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat(
        [torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0
    )
    outputs["pred"] = [p for p, m in zip(outputs["pred"], shifted_reset_mask) if not m]
    outputs["views"] = [v for v, m in zip(outputs["views"], shifted_reset_mask) if not m]
    reset_mask = reset_mask[~shifted_reset_mask]

    assert len(outputs["pred"]) == len(img_paths), \
        f"Preds {len(outputs['pred'])} != images {len(img_paths)}"

    pts3ds_self = torch.cat(
        [o["pts3d_in_self_view"].cpu() for o in outputs["pred"]], 0
    )
    B, Hm, Wm, _ = pts3ds_self.shape

    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    if reset_mask.any():
        identity = torch.eye(4, device=pr_poses[0].device)
        reset_poses = torch.where(
            reset_mask.unsqueeze(-1).unsqueeze(-1),
            torch.cat(pr_poses, 0), identity
        )
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat(
            [identity.unsqueeze(0), cumulative_bases[:-1]], dim=0
        )
        all_poses = torch.einsum('bij,bjk->bik', shifted_bases,
                                 torch.cat(pr_poses, 0))
        pr_poses = list(all_poses.unsqueeze(1).unbind(0))

    pp = torch.tensor([Wm // 2, Hm // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal_model = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    scale_x = orig_w / Wm
    scale_y = orig_h / Hm

    fx_orig = focal_model.cpu().numpy() * scale_x
    fy_orig = focal_model.cpu().numpy() * scale_y
    cx_orig = (Wm / 2) * scale_x
    cy_orig = (Hm / 2) * scale_y

    for idx, (img_path, pose) in enumerate(zip(img_paths, pr_poses)):
        img_name = os.path.basename(img_path)
        stem = os.path.splitext(img_name)[0]
        c2w = pose.squeeze(0).cpu().numpy()

        camera_data = {
            "image_name": img_name,
            "width": orig_w,
            "height": orig_h,
            "fx": float(fx_orig[idx]),
            "fy": float(fy_orig[idx]),
            "cx": float(cx_orig),
            "cy": float(cy_orig),
            "camera_to_world": c2w.tolist(),
        }

        json_path = os.path.join(scene_dir, f"{stem}_camera.json")
        with open(json_path, "w") as f:
            json.dump(camera_data, f, indent=2)

    print(f"  Saved {len(img_paths)} camera JSON files")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--image_sets_dir", type=str,
                        default="/root/autodl-tmp/datasets/image_sets")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_update_type", type=str, default="ttt3r")
    parser.add_argument("--reset_interval", type=int, default=1000000)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    add_path_to_dust3r(args.model_path)

    from src.dust3r.model import ARCroco3DStereo

    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    scene_dirs = sorted([
        d for d in glob.glob(os.path.join(args.image_sets_dir, '*'))
        if os.path.isdir(d)
    ])

    print(f"Found {len(scene_dirs)} scenes:")
    for d in scene_dirs:
        print(f"  - {os.path.basename(d)}")

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        print(f"\n{'='*55}\n[{scene_name}]\n{'='*55}")
        run_ttt3r_on_scene(
            scene_dir, model, device,
            size=args.size,
            reset_interval=args.reset_interval
        )

    print(f"\n{'='*55}")
    print("All scenes processed.")


if __name__ == "__main__":
    main()
