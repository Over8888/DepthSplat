#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_OUT_ROOT = Path("/root/autodl-tmp/outputs/ttt3r_pmr_chain")
DEFAULT_RE10K_ROOT = Path("/root/autodl-tmp/RealEstate10K")
DEFAULT_EVAL_INDEX = Path("/root/depthsplat/assets/evaluation_index_re10k_video.json")
DEFAULT_TTT3R_ROOT = Path("/root/TTT3R")
DEFAULT_TTT3R_MODEL = Path("/root/autodl-tmp/cut3r_512_dpt_4_64.pth")
DEFAULT_TTT3R_PYTHON = Path("/root/miniconda3/envs/depthsplat/bin/python")
DEFAULT_DEPTHSPLAT_PYTHON = Path("/root/miniconda3/envs/depthsplat/bin/python")
DEFAULT_LARGE_CKPT = Path("/root/depthsplat/pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth")
DEBUG_LOG = Path("/root/autodl-tmp/outputs/ttt3r_pmr_chain_prepare_debug.log")


def debug(message: str) -> None:
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
        f.flush()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def choose_scene(re10k_root: Path, eval_index_path: Path, scene: str | None) -> tuple[str, dict, Path]:
    eval_index = read_json(eval_index_path)
    dataset_index = read_json(re10k_root / "test" / "index.json")
    candidates = [scene] if scene else list(eval_index.keys())
    for key in candidates:
        entry = eval_index.get(key)
        chunk_name = dataset_index.get(key)
        if entry is None or chunk_name is None:
            continue
        chunk_path = re10k_root / "test" / chunk_name
        if chunk_path.exists():
            return key, entry, chunk_path
    raise RuntimeError("No usable scene found in both evaluation index and RE10K test index")


def load_scene_from_chunk(chunk_path: Path, scene_key: str) -> dict:
    import torch

    chunk = torch.load(chunk_path, map_location="cpu")
    matches = [example for example in chunk if example["key"] == scene_key]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one scene {scene_key} in {chunk_path}, got {len(matches)}")
    return matches[0]


def decode_image(image_tensor: Any, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(BytesIO(image_tensor.numpy().tobytes())).convert("RGB")
    image.save(output_path)
    return image.size


def prepare_inputs(args: argparse.Namespace) -> None:
    debug("prepare:start")
    out_root = args.out_root
    scene_key, entry, chunk_path = choose_scene(args.re10k_root, args.eval_index, args.scene)
    debug(f"prepare:chosen scene={scene_key} chunk={chunk_path}")
    example = load_scene_from_chunk(chunk_path, scene_key)
    debug(f"prepare:loaded scene={scene_key}")

    frame_indices = sorted(set(entry["context"]) | set(entry["target"]))
    debug(f"prepare:frames count={len(frame_indices)}")
    image_dir = out_root / "input_images" / scene_key
    if image_dir.exists() and args.clean:
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    image_sizes = {}
    for index in frame_indices:
        path = image_dir / f"{index:06d}.jpg"
        width, height = decode_image(example["images"][index], path)
        image_paths.append(str(path))
        image_sizes[str(index)] = {"width": width, "height": height}
    debug(f"prepare:decoded images={len(image_paths)}")

    metadata = {
        "scene": scene_key,
        "chunk_path": str(chunk_path),
        "context": entry["context"],
        "target": entry["target"],
        "frame_indices": frame_indices,
        "image_dir": str(image_dir),
        "image_paths": image_paths,
        "image_sizes": image_sizes,
        "original_re10k_cameras_18d": np.asarray(example["cameras"], dtype=np.float32).tolist(),
    }
    write_json(out_root / "metadata" / "re10k_sample_metadata.json", metadata)
    debug("prepare:wrote metadata")
    print(json.dumps({"scene": scene_key, "frames": len(frame_indices), "image_dir": str(image_dir)}, ensure_ascii=False))


def run_ttt3r(args: argparse.Namespace) -> None:
    if not args.ttt3r_model_path.exists():
        raise RuntimeError(f"TTT3R checkpoint not found: {args.ttt3r_model_path}")
    metadata = read_json(args.out_root / "metadata" / "re10k_sample_metadata.json")
    request = {
        "image_paths": metadata["image_paths"],
        "ttt3r_root": str(args.ttt3r_root),
        "model_path": str(args.ttt3r_model_path),
        "size": args.ttt3r_size,
        "model_update_type": args.ttt3r_model_update_type,
        "reset_interval": args.ttt3r_reset_interval,
        "device": args.ttt3r_device,
    }
    request_path = args.out_root / "ttt3r" / "request.json"
    output_path = args.out_root / "ttt3r" / "result.npz"
    meta_output_path = args.out_root / "ttt3r" / "result_meta.json"
    write_json(request_path, request)

    command = [
        str(args.ttt3r_python),
        "-m",
        "app.services.ttt3r_subprocess",
        "--request",
        str(request_path),
        "--output",
        str(output_path),
        "--meta-output",
        str(meta_output_path),
    ]
    env = dict(**os_environ_with_pythonpath(Path("/root/autodl-tmp/demo_v2"), args.ttt3r_root / "src"))
    result = subprocess.run(command, cwd="/root/autodl-tmp/demo_v2", env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (args.out_root / "ttt3r" / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (args.out_root / "ttt3r" / "stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"TTT3R subprocess failed with exit code {result.returncode}; see {args.out_root / 'ttt3r'}")
    print(json.dumps({"ttt3r_result": str(output_path), "ttt3r_meta": str(meta_output_path)}, ensure_ascii=False))


def os_environ_with_pythonpath(*extra_paths: Path) -> dict[str, str]:
    import os

    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    prefix = ":".join(str(path) for path in extra_paths)
    env["PYTHONPATH"] = prefix if not current else f"{prefix}:{current}"
    return env


def camera_18d_from_ttt3r(intrinsics_px: np.ndarray, extrinsics_w2c: np.ndarray, width: int, height: int) -> np.ndarray:
    cameras = np.zeros((intrinsics_px.shape[0], 18), dtype=np.float32)
    cameras[:, 0] = intrinsics_px[:, 0, 0] / width
    cameras[:, 1] = intrinsics_px[:, 1, 1] / height
    cameras[:, 2] = intrinsics_px[:, 0, 2] / width
    cameras[:, 3] = intrinsics_px[:, 1, 2] / height
    cameras[:, 6:] = extrinsics_w2c.reshape(extrinsics_w2c.shape[0], 12)
    return cameras


def build_dataset(args: argparse.Namespace) -> None:
    import torch

    metadata = read_json(args.out_root / "metadata" / "re10k_sample_metadata.json")
    scene = metadata["scene"]
    frame_indices = metadata["frame_indices"]
    first_size = metadata["image_sizes"][str(frame_indices[0])]
    width = int(first_size["width"])
    height = int(first_size["height"])

    ttt3r_npz_path = args.out_root / "ttt3r" / "result.npz"
    if not ttt3r_npz_path.exists():
        raise RuntimeError(f"TTT3R result not found: {ttt3r_npz_path}")
    with np.load(ttt3r_npz_path) as payload:
        extrinsics_w2c = np.asarray(payload["extrinsics_w2c"], dtype=np.float32)
        intrinsics_px = np.asarray(payload["intrinsics_px"], dtype=np.float32)

    if len(frame_indices) != intrinsics_px.shape[0] or len(frame_indices) != extrinsics_w2c.shape[0]:
        raise RuntimeError("TTT3R camera count does not match prepared frame count")

    ttt3r_cameras = camera_18d_from_ttt3r(intrinsics_px, extrinsics_w2c, width, height)
    original = np.asarray(metadata["original_re10k_cameras_18d"], dtype=np.float32)
    new_cameras = original.copy()
    for local_i, frame_i in enumerate(frame_indices):
        new_cameras[frame_i] = ttt3r_cameras[local_i]

    source_example = load_scene_from_chunk(Path(metadata["chunk_path"]), scene)
    out_example = {
        "key": scene,
        "images": source_example["images"],
        "cameras": torch.from_numpy(new_cameras),
    }
    dataset_test = args.out_root / "dataset" / "test"
    dataset_test.mkdir(parents=True, exist_ok=True)
    torch.save([out_example], dataset_test / "000000.torch")
    write_json(dataset_test / "index.json", {scene: "000000.torch"})
    write_json(args.out_root / "evaluation_index.json", {scene: {"context": metadata["context"], "target": metadata["target"]}})
    write_json(args.out_root / "metadata" / "ttt3r_camera_conversion.json", {
        "scene": scene,
        "width": width,
        "height": height,
        "frame_indices": frame_indices,
        "ttt3r_result": str(ttt3r_npz_path),
        "dataset_root": str(args.out_root / "dataset"),
        "evaluation_index": str(args.out_root / "evaluation_index.json"),
    })
    print(json.dumps({"dataset_root": str(args.out_root / "dataset"), "evaluation_index": str(args.out_root / "evaluation_index.json")}, ensure_ascii=False))


def print_depthsplat_command(args: argparse.Namespace) -> None:
    out = args.out_root / "depthsplat_large_pmr_smooth"
    command = [
        str(args.depthsplat_python), "-m", "src.main", "+experiment=re10k", "mode=test",
        f"dataset.roots=[{args.out_root / 'dataset'}]",
        "dataset/view_sampler=evaluation",
        f"dataset.view_sampler.index_path={args.out_root / 'evaluation_index.json'}",
        "dataset.view_sampler.num_context_views=2",
        "dataset.image_shape=[256,256]",
        "dataset.skip_bad_shape=false",
        "data_loader.test.num_workers=0",
        "wandb.mode=disabled",
        "trainer.num_sanity_val_steps=0",
        f"checkpointing.pretrained_model={args.large_checkpoint}",
        "model.encoder.monodepth_vit_type=vitl",
        "model.encoder.num_scales=2",
        "model.encoder.upsample_factor=2",
        "model.encoder.lowest_feature_resolution=4",
        "model.encoder.cost_volume_confidence=true",
        "model.encoder.pmr_guided_smooth=true",
        "test.compute_scores=true",
        "test.save_image=false",
        "test.save_video=false",
        "test.save_depth=false",
        "test.save_gaussian=false",
        f"test.output_path={out / 'metrics'}",
        f"output_dir={out}",
    ]
    print(" ".join(str(part) for part in command))


def summarize(args: argparse.Namespace) -> None:
    metrics_dir = args.out_root / "depthsplat_large_pmr_smooth" / "metrics"
    paths = {
        "scores_all_avg": metrics_dir / "scores_all_avg.json",
        "scores_per_scene": metrics_dir / "scores_per_scene.json",
        "benchmark": metrics_dir / "benchmark.json",
        "peak_memory": metrics_dir / "peak_memory.json",
    }
    summary = {}
    for key, path in paths.items():
        summary[key] = read_json(path) if path.exists() else None
    write_json(args.out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RE10K -> TTT3R -> PMR-guided DepthSplat chain experiment helpers.")
    parser.add_argument("stage", choices=["prepare", "ttt3r", "build-dataset", "print-eval-command", "summarize"])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--re10k-root", type=Path, default=DEFAULT_RE10K_ROOT)
    parser.add_argument("--eval-index", type=Path, default=DEFAULT_EVAL_INDEX)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--ttt3r-root", type=Path, default=DEFAULT_TTT3R_ROOT)
    parser.add_argument("--ttt3r-python", type=Path, default=DEFAULT_TTT3R_PYTHON if DEFAULT_TTT3R_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--ttt3r-model-path", type=Path, default=DEFAULT_TTT3R_MODEL)
    parser.add_argument("--ttt3r-size", type=int, default=512)
    parser.add_argument("--ttt3r-model-update-type", type=str, default="ttt3r")
    parser.add_argument("--ttt3r-reset-interval", type=int, default=200)
    parser.add_argument("--ttt3r-device", type=str, default="cuda")
    parser.add_argument("--depthsplat-python", type=Path, default=DEFAULT_DEPTHSPLAT_PYTHON if DEFAULT_DEPTHSPLAT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--large-checkpoint", type=Path, default=DEFAULT_LARGE_CKPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "prepare":
        prepare_inputs(args)
    elif args.stage == "ttt3r":
        run_ttt3r(args)
    elif args.stage == "build-dataset":
        build_dataset(args)
    elif args.stage == "print-eval-command":
        print_depthsplat_command(args)
    elif args.stage == "summarize":
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())