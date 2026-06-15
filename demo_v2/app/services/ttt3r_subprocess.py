from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TTT3R camera estimation in an isolated subprocess.")
    parser.add_argument("--request", type=Path, required=True, help="Path to the input request JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output NPZ file.")
    parser.add_argument("--meta-output", type=Path, required=True, help="Path to the output metadata JSON file.")
    return parser.parse_args()


def _load_ttt3r_modules(ttt3r_root: Path, model_path: Path):
    root_str = str(ttt3r_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    cwd = Path.cwd()
    os.chdir(ttt3r_root)
    try:
        from add_ckpt_path import add_path_to_dust3r

        add_path_to_dust3r(str(model_path))
        from demo import prepare_input
        from src.dust3r.inference import inference_recurrent_lighter
        from src.dust3r.model import ARCroco3DStereo
    finally:
        os.chdir(cwd)

    return prepare_input, inference_recurrent_lighter, ARCroco3DStereo


def _load_camera_outputs(camera_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    camera_paths = sorted(camera_dir.glob("*.npz"))
    if not camera_paths:
        raise RuntimeError(f"TTT3R produced no camera files in {camera_dir}")

    c2w_list: list[np.ndarray] = []
    intrinsics_list: list[np.ndarray] = []
    for path in camera_paths:
        payload = np.load(path)
        c2w_list.append(np.asarray(payload["pose"], dtype=np.float32))
        intrinsics_list.append(np.asarray(payload["intrinsics"], dtype=np.float32))
    return np.stack(c2w_list), np.stack(intrinsics_list)


def _rescale_intrinsics_to_input_images(
    intrinsics_px: np.ndarray,
    ttt3r_output_dir: Path,
    image_paths: list[str],
) -> np.ndarray:
    color_paths = sorted((ttt3r_output_dir / "color").glob("*.png"))
    if not color_paths:
        raise RuntimeError(f"TTT3R produced no color images in {ttt3r_output_dir / 'color'}")

    with Image.open(color_paths[0]) as image:
        processed_width, processed_height = image.size
    with Image.open(image_paths[0]) as image:
        input_width, input_height = image.size

    scale_x = float(input_width) / float(processed_width)
    scale_y = float(input_height) / float(processed_height)
    scaled = intrinsics_px.copy()
    scaled[:, 0, 0] *= scale_x
    scaled[:, 0, 2] *= scale_x
    scaled[:, 1, 1] *= scale_y
    scaled[:, 1, 2] *= scale_y
    return scaled.astype(np.float32)


def _extract_camera_outputs(outputs: dict, image_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.geometry import matrix_cumprod

    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    preds = [pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self = torch.cat([pred["pts3d_in_self_view"].cpu() for pred in preds], 0)
    pr_poses = [pose_encoding_to_camera(pred["camera_pose"].clone()).cpu() for pred in preds]

    if reset_mask.any():
        poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        poses = torch.einsum("bij,bjk->bik", shifted_bases, poses)
    else:
        poses = torch.cat(pr_poses, 0)

    batch, height, width, _ = pts3ds_self.shape
    pp = torch.tensor([width // 2, height // 2], device=pts3ds_self.device).float().repeat(batch, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld").detach().cpu()

    intrinsics_px = torch.eye(3).unsqueeze(0).repeat(poses.shape[0], 1, 1)
    intrinsics_px[:, 0, 0] = focal
    intrinsics_px[:, 1, 1] = focal
    intrinsics_px[:, 0, 2] = pp[:, 0].cpu()
    intrinsics_px[:, 1, 2] = pp[:, 1].cpu()

    with Image.open(image_paths[0]) as image:
        input_width, input_height = image.size
    intrinsics_px[:, 0, 0] *= float(input_width) / float(width)
    intrinsics_px[:, 0, 2] *= float(input_width) / float(width)
    intrinsics_px[:, 1, 1] *= float(input_height) / float(height)
    intrinsics_px[:, 1, 2] *= float(input_height) / float(height)

    return poses.numpy().astype(np.float32), intrinsics_px.numpy().astype(np.float32)


def _validate_cameras(c2w: np.ndarray, intrinsics_px: np.ndarray, expected_count: int) -> None:
    if c2w.shape != (expected_count, 4, 4):
        raise RuntimeError(f"Expected {expected_count} TTT3R poses, got shape {c2w.shape}")
    if intrinsics_px.shape != (expected_count, 3, 3):
        raise RuntimeError(f"Expected {expected_count} TTT3R intrinsics, got shape {intrinsics_px.shape}")
    if not np.isfinite(c2w).all() or not np.isfinite(intrinsics_px).all():
        raise RuntimeError("TTT3R camera outputs contain NaN or Inf")
    if (intrinsics_px[:, 0, 0] <= 0).any() or (intrinsics_px[:, 1, 1] <= 0).any():
        raise RuntimeError("TTT3R camera outputs contain non-positive focal lengths")


def _run(request_payload: dict, output_path: Path, meta_output_path: Path) -> None:
    import torch

    image_paths = [str(path) for path in request_payload["image_paths"]]
    if not image_paths:
        raise ValueError("At least one image is required for TTT3R camera estimation")

    ttt3r_root = Path(request_payload["ttt3r_root"])
    model_path = Path(request_payload["model_path"])
    if not ttt3r_root.exists():
        raise RuntimeError(f"TTT3R repository not found: {ttt3r_root}")
    if not model_path.exists():
        raise RuntimeError(f"TTT3R checkpoint not found: {model_path}")

    prepare_input, inference_recurrent_lighter, model_cls = _load_ttt3r_modules(ttt3r_root, model_path)

    device = str(request_payload.get("device") or "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TTT3R requested CUDA but CUDA is not available")

    views = prepare_input(
        img_paths=image_paths,
        img_mask=[True] * len(image_paths),
        size=int(request_payload["size"]),
        revisit=1,
        update=True,
        reset_interval=int(request_payload["reset_interval"]),
    )

    model = model_cls.from_pretrained(str(model_path)).to(device)
    model.config.model_update_type = str(request_payload["model_update_type"])
    model.eval()

    with torch.inference_mode():
        outputs, _ = inference_recurrent_lighter(views, model, device)

    c2w, intrinsics_px = _extract_camera_outputs(outputs, image_paths)

    _validate_cameras(c2w, intrinsics_px, len(image_paths))
    # DepthSplat stores OpenCV-style world-to-camera extrinsics as 3x4 rows
    # inside its 18D camera vector. Keep only the affine camera rows here.
    w2c = np.linalg.inv(c2w).astype(np.float32)[:, :3, :4]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, extrinsics_w2c=w2c, intrinsics_px=intrinsics_px)
    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_output_path.write_text(
        json.dumps(
            {
                "device": device,
                "source": str(model_path),
                "image_count": len(image_paths),
                "model_update_type": str(request_payload["model_update_type"]),
                "reset_interval": int(request_payload["reset_interval"]),
                "size": int(request_payload["size"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    del model, outputs, views
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    args = _parse_args()
    request_payload = json.loads(args.request.read_text(encoding="utf-8"))
    _run(request_payload, args.output, args.meta_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
