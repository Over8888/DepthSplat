from __future__ import annotations

import argparse
import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT inference in an isolated subprocess.")
    parser.add_argument("--request", type=Path, required=True, help="Path to the input request JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output NPZ file.")
    parser.add_argument("--meta-output", type=Path, required=True, help="Path to the output metadata JSON file.")
    return parser.parse_args()


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"VGGT device '{requested}' was requested but CUDA is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    capability = torch.cuda.get_device_capability(device)
    if capability[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def _resolve_checkpoint_path(request_payload: dict) -> Path | None:
    explicit = request_payload.get("checkpoint_path")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
        raise RuntimeError(f"Configured VGGT checkpoint not found: {path}")
    return None


def _load_state_dict(checkpoint_path: Path) -> dict:
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            nested = state_dict.get(key)
            if isinstance(nested, dict):
                state_dict = nested
                break
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported VGGT checkpoint format: {checkpoint_path}")

    normalized = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model.") :]
        normalized[clean_key] = value
    return normalized


def _rescale_intrinsics_to_original(
    intrinsics: np.ndarray,
    original_coords: np.ndarray,
    inference_resolution: int,
) -> np.ndarray:
    scaled = intrinsics.copy()
    resolution = float(inference_resolution)
    real_widths = original_coords[:, 4]
    real_heights = original_coords[:, 5]
    resize_ratios = np.maximum(real_widths, real_heights) / resolution

    scaled[:, 0, 0] *= resize_ratios
    scaled[:, 1, 1] *= resize_ratios
    scaled[:, 0, 2] = real_widths / 2.0
    scaled[:, 1, 2] = real_heights / 2.0
    scaled[:, 2, 2] = 1.0
    return scaled.astype(np.float32)


def _ensure_repo_modules(vggt_root: Path):
    repo_root_str = str(vggt_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    try:
        load_module = importlib.import_module("vggt.utils.load_fn")
        pose_module = importlib.import_module("vggt.utils.pose_enc")
        vggt_cls = importlib.import_module("vggt.models.vggt").VGGT
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "unknown"
        raise RuntimeError(
            f"Failed to import VGGT dependency '{missing_name}'. Install VGGT runtime dependencies before using uploaded-image tasks."
        ) from exc
    return load_module.load_and_preprocess_images_square, pose_module.pose_encoding_to_extri_intri, vggt_cls


def _run(request_payload: dict, output_path: Path, meta_output_path: Path) -> None:
    image_paths = request_payload["image_paths"]
    if not image_paths:
        raise ValueError("At least one image is required for VGGT camera estimation")

    vggt_root = Path(request_payload["vggt_root"])
    if not vggt_root.exists():
        raise RuntimeError(f"VGGT repository not found: {vggt_root}")

    load_fn, pose_fn, vggt_cls = _ensure_repo_modules(vggt_root)
    device = _resolve_device(request_payload.get("device"))
    dtype = _resolve_dtype(device)
    checkpoint_path = _resolve_checkpoint_path(request_payload)

    if checkpoint_path is not None:
        model = vggt_cls()
        state_dict = _load_state_dict(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)
        load_source = str(checkpoint_path)
    else:
        model = vggt_cls.from_pretrained(request_payload["model_id"])
        load_source = request_payload["model_id"]

    model.eval()
    model = model.to(device)

    images, original_coords = load_fn(image_paths, int(request_payload["load_resolution"]))
    images = images.to(device)

    with torch.inference_mode():
        with _autocast_context(device, dtype):
            resized = F.interpolate(
                images,
                size=(int(request_payload["inference_resolution"]), int(request_payload["inference_resolution"])),
                mode="bilinear",
                align_corners=False,
            )
            aggregated_tokens_list, _ = model.aggregator(resized[None])
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsics, intrinsics = pose_fn(pose_enc, resized.shape[-2:])

    extrinsics_np = extrinsics.squeeze(0).detach().cpu().float().numpy()
    intrinsics_np = intrinsics.squeeze(0).detach().cpu().float().numpy()
    coords_np = original_coords.detach().cpu().float().numpy()
    intrinsics_px = _rescale_intrinsics_to_original(
        intrinsics_np,
        coords_np,
        int(request_payload["inference_resolution"]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, extrinsics_w2c=extrinsics_np, intrinsics_px=intrinsics_px)
    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_output_path.write_text(
        json.dumps(
            {
                "device": str(device),
                "source": load_source,
                "image_count": len(image_paths),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    del model, images, resized, aggregated_tokens_list, pose_enc, extrinsics, intrinsics
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    args = _parse_args()
    request_payload = json.loads(args.request.read_text(encoding="utf-8"))
    _run(request_payload, args.output, args.meta_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
