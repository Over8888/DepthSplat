from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.config import PresetConfig, Settings
from app.models.sample import SampleRecord
from app.services.storage import FilesystemStorage
from app.services.ttt3r_service import TTT3RService
from app.services.vggt_service import VGGTService
from app.utils.logging import get_logger
from app.utils.timezone import now_local

_DEPTHSPLAT_SRC_INFERENCE = None


def _pose_loader():
    global _DEPTHSPLAT_SRC_INFERENCE
    if _DEPTHSPLAT_SRC_INFERENCE is None:
        import sys
        depthsplat_root = str(Path("/root/depthsplat"))
        inference_path = str(Path(depthsplat_root) / "src" / "inference")
        if inference_path not in sys.path:
            sys.path.insert(0, inference_path)
        from pose_loader import load_poses, load_image_to_tensor  # type: ignore[import-untyped]
        _DEPTHSPLAT_SRC_INFERENCE = (load_poses, load_image_to_tensor)
    return _DEPTHSPLAT_SRC_INFERENCE


class SampleService:
    def __init__(
        self,
        settings: Settings,
        storage: FilesystemStorage,
        vggt_service: VGGTService,
        ttt3r_service: TTT3RService,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.vggt_service = vggt_service
        self.ttt3r_service = ttt3r_service
        self.logger = get_logger(__name__)

    def list_presets(self) -> list[PresetConfig]:
        return list(self.settings.presets.values())

    @lru_cache(maxsize=32)
    def list_samples(self, preset_name: str) -> list[SampleRecord]:
        preset = self._preset(preset_name)
        index_path = preset.dataset_root / preset.sample_stage / "index.json"
        if not index_path.exists():
            return []
        dataset_index = json.loads(index_path.read_text(encoding="utf-8"))
        eval_index = json.loads(preset.fixed_index_path.read_text(encoding="utf-8"))
        scene_keys = self._select_sample_scene_keys(preset, dataset_index, eval_index)
        samples: list[SampleRecord] = []
        for key in scene_keys:
            entry = eval_index.get(key, {})
            context_indices = list(entry.get("context", []))
            target_indices = list(entry.get("target", []))
            input_view_count = len(context_indices) or preset.num_context_views
            label = f"{key} ({preset.display_name})"
            image_shape = list(preset.image_shape)
            quality = "large" if "large" in preset.name else "small" if "small" in preset.name else "base"
            category = "dl3dv" if "dl3dv" in str(preset.dataset_root).lower() or preset.name.startswith("re10k_base_4view") or preset.name.startswith("re10k_base_6view") else "re10k"
            source_chunk = (preset.dataset_root / preset.sample_stage / dataset_index[key]).resolve()
            thumbnail_url = self._ensure_sample_preview(preset, key, source_chunk)
            samples.append(
                SampleRecord(
                    id=f"{preset.name}:{key}",
                    preset=preset.name,
                    scene_key=key,
                    label=label,
                    name=label,
                    thumbnail_url=thumbnail_url,
                    preview_images=[thumbnail_url] if thumbnail_url else [],
                    description=(
                        f"{preset.display_name}; {input_view_count} input views; "
                        f"{image_shape[0]}x{image_shape[1]} model resolution"
                    ),
                    category=category,
                    tags=[quality, f"{input_view_count}-view", f"{image_shape[0]}x{image_shape[1]}", category],
                    scene_number=key,
                    input_view_count=input_view_count,
                    target_view_count=len(target_indices),
                    source_chunk=source_chunk,
                    defaults={
                        "checkpoint": preset.checkpoint_path.name,
                        "num_context_views": preset.num_context_views,
                        "image_shape": image_shape,
                        "input_view_count": input_view_count,
                        "target_view_count": len(target_indices),
                    },
                )
            )
        return samples

    def _ensure_sample_preview(self, preset: PresetConfig, scene_key: str, source_chunk: Path) -> str | None:
        preview_name = hashlib.sha256(scene_key.encode("utf-8")).hexdigest() + ".jpg"
        relative_path = Path("_sample_previews") / preset.name / preview_name
        preview_path = self.settings.outputs_root / relative_path
        if preview_path.exists():
            return "/artifacts/" + relative_path.as_posix()

        try:
            chunk = torch.load(source_chunk, map_location="cpu")
            example = next(item for item in chunk if item["key"] == scene_key)
            image_bytes = example["images"][0].numpy().tobytes()
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(BytesIO(image_bytes)) as image:
                image.convert("RGB").save(preview_path, format="JPEG", quality=85, optimize=True)
            return "/artifacts/" + relative_path.as_posix()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "sample preview generation failed",
                event="sample_preview_failed",
                fields={"preset": preset.name, "scene_key": scene_key, "error": str(exc)},
            )
            return None

    def _select_sample_scene_keys(self, preset: PresetConfig, dataset_index: dict, eval_index: dict) -> list[str]:
        available_keys = sorted(set(dataset_index).intersection(eval_index))
        if not available_keys:
            return []

        source_group = (preset.dataset_root.resolve(), preset.sample_stage)
        sibling_presets = [
            item
            for item in self.settings.presets.values()
            if (item.dataset_root.resolve(), item.sample_stage) == source_group
        ]
        sibling_presets.sort(key=lambda item: (item.num_context_views, item.name))
        sibling_names = [item.name for item in sibling_presets]
        try:
            group_index = sibling_names.index(preset.name)
        except ValueError:
            group_index = 0

        limit = preset.sample_limit
        start = group_index * limit
        if start + limit <= len(available_keys):
            return available_keys[start : start + limit]

        if len(available_keys) >= limit:
            wrapped = available_keys[start % len(available_keys) :] + available_keys[: start % len(available_keys)]
            return wrapped[:limit]

        return available_keys

    def get_sample(self, sample_id: str) -> SampleRecord:
        preset_name, scene_key = self.parse_sample_id(sample_id)
        for sample in self.list_samples(preset_name):
            if sample.scene_key == scene_key:
                return sample
        raise ValueError(f"Unknown sample_id: {sample_id}")

    def parse_sample_id(self, sample_id: str) -> tuple[str, str]:
        if ":" not in sample_id:
            raise ValueError("sample_id must have format <preset>:<scene_key>")
        return sample_id.split(":", 1)

    def materialize_sample(self, task_dir: Path, sample_id: str, preset_name: str, input_view_count: int | None = None) -> dict:
        preset = self._preset(preset_name)
        sample = self.get_sample(sample_id)
        if sample.preset != preset_name:
            raise ValueError("sample_id does not match preset")
        chunk = torch.load(sample.source_chunk)
        matches = [item for item in chunk if item["key"] == sample.scene_key]
        if not matches:
            raise ValueError(f"Scene {sample.scene_key} not found in chunk {sample.source_chunk}")
        example = matches[0]
        context_indices, target_indices = self._resolve_eval_indices(preset, sample.scene_key)

        if input_view_count is not None and input_view_count != len(context_indices):
            raise ValueError(
                "inputViewCount does not match sample evaluation index: "
                f"got {input_view_count}, expected {len(context_indices)} for {preset.name}"
            )

        dataset_root = task_dir / "input" / "dataset" / "test"
        dataset_root.mkdir(parents=True, exist_ok=True)
        torch.save([example], dataset_root / "000000.torch")
        (dataset_root / "index.json").write_text(json.dumps({sample.scene_key: "000000.torch"}, indent=2), encoding="utf-8")

        eval_meta_path = task_dir / "meta" / "evaluation_index.json"
        eval_meta_path.write_text(
            json.dumps({sample.scene_key: {"context": context_indices, "target": target_indices}}, indent=2),
            encoding="utf-8",
        )

        preview_dir = task_dir / "input" / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_files: list[str] = []
        for order, image_index in enumerate(context_indices):
            image = Image.open(BytesIO(bytes(example["images"][image_index].tolist())))
            filename = f"context_{order:02d}_{image_index:06d}.png"
            image.save(preview_dir / filename)
            preview_files.append(f"input/preview/{filename}")

        source_height, source_width = self._source_image_shape(preset)
        cameras_np = example["cameras"].numpy() if isinstance(example["cameras"], torch.Tensor) else np.array(example["cameras"])
        frames_camera_data = self._deserialize_cameras_from_18d(cameras_np, source_width, source_height)
        image_height, image_width = preset.image_shape
        self._save_camera_params(task_dir, frames_camera_data, image_width, image_height)

        return {
            "scene_key": sample.scene_key,
            "dataset_root": dataset_root.parent,
            "evaluation_index_path": eval_meta_path,
            "input_preview_files": preview_files,
            "context_indices": context_indices,
            "target_indices": target_indices,
            "target_count": len(target_indices),
        }

    def materialize_uploaded_images(
        self,
        task_dir: Path,
        image_paths: list[str],
        video_path: str | None,
        scene_key: str,
        preset_name: str,
        requested_context_indices: list[int],
    ) -> dict:
        preset = self._preset(preset_name)
        source_height, source_width = self._source_image_shape(preset)
        use_ttt3r_video_strategy = self._should_use_ttt3r_video_strategy(
            preset,
            video_path,
            requested_context_indices,
        )
        preview_dir = task_dir / "input" / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        dataset_root = task_dir / "input" / "dataset" / "test"
        dataset_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix=f"depthsplat_v3_{scene_key}_"))
        prepared_dir = work_dir / "prepared_frames"
        prepared_dir.mkdir(parents=True, exist_ok=True)

        try:
            source_paths = self._resolve_source_paths(
                task_dir,
                work_dir,
                image_paths,
                video_path,
                preset,
                requested_context_indices,
            )
            if not use_ttt3r_video_strategy:
                source_paths = self._trim_dark_video_edges(source_paths, preset, requested_context_indices, video_path, task_dir)
            if len(source_paths) < preset.expected_min_views:
                raise ValueError(
                    f"Preset {preset.display_name} requires at least {preset.expected_min_views} frames, got {len(source_paths)}"
                )

            if use_ttt3r_video_strategy:
                context_indices = self._select_context_indices_floor(len(source_paths), preset.num_context_views)
                target_indices = list(range(len(source_paths)))
                retained_source_paths = source_paths
                self._append_task_log(
                    task_dir,
                    "stdout",
                    (
                        "[video] using TTT3R contiguous-clip strategy with "
                        f"{len(retained_source_paths)} consecutive frames"
                    ),
                )
            else:
                context_indices = self._resolve_context_indices(requested_context_indices, len(source_paths), preset.num_context_views)
                target_indices = list(range(min(context_indices), max(context_indices) + 1))
                retained_source_paths = source_paths[: target_indices[-1] + 1]

            camera_backend = self._camera_backend_for_upload(preset, video_path, requested_context_indices)
            self._append_task_log(
                task_dir,
                "stdout",
                f"[upload] preparing {len(retained_source_paths)} frames for {camera_backend.upper()}",
            )

            prepared_paths: list[Path] = []
            encoded_images: list[torch.Tensor] = []
            preview_files: list[str] = []
            context_index_set = set(context_indices)
            for order, source_path in enumerate(retained_source_paths):
                filename = f"frame_{order:04d}.png"
                if use_ttt3r_video_strategy:
                    prepared_path = source_path
                    prepared_paths.append(prepared_path)
                    if order in context_index_set:
                        preview_path = preview_dir / filename
                        shutil.copy2(prepared_path, preview_path)
                        preview_files.append(f"input/preview/{filename}")
                    encoded_images.append(self._encode_image_file(prepared_path))
                else:
                    with Image.open(source_path) as image:
                        prepared = self._prepare_uploaded_image(image, (source_height, source_width))
                        prepared_path = prepared_dir / filename
                        prepared.save(prepared_path)
                        prepared_paths.append(prepared_path)
                        if order in context_index_set:
                            preview_path = preview_dir / filename
                            prepared.save(preview_path)
                            preview_files.append(f"input/preview/{filename}")
                        encoded_images.append(self._encode_image(prepared))

            self._append_task_log(
                task_dir,
                "stdout",
                f"[upload] running {camera_backend.upper()} camera estimation on {len(prepared_paths)} frames",
            )
            if camera_backend == "ttt3r":
                prediction = self.ttt3r_service.predict_cameras(prepared_paths)
            else:
                prediction = self.vggt_service.predict_cameras(prepared_paths)

            manual_images_dir, manual_poses_path, manual_target_poses_path = self._write_manual_inputs(
                task_dir,
                prepared_paths,
                prediction.extrinsics_w2c,
                prediction.intrinsics_px,
                source_width,
                source_height,
                context_indices=context_indices,
                target_indices=target_indices,
            )
            cameras = self._serialize_cameras(prediction.extrinsics_w2c, prediction.intrinsics_px, source_width, source_height)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        example = {
            "url": "upload",
            "timestamps": torch.arange(len(prepared_paths), dtype=torch.int64),
            "cameras": torch.tensor(np.asarray(cameras), dtype=torch.float32),
            "images": encoded_images,
            "key": scene_key,
        }
        torch.save([example], dataset_root / "000000.torch")
        (dataset_root / "index.json").write_text(json.dumps({scene_key: "000000.torch"}, indent=2), encoding="utf-8")

        eval_meta_path = task_dir / "meta" / "evaluation_index.json"
        eval_meta_path.write_text(
            json.dumps({scene_key: {"context": context_indices, "target": target_indices}}, indent=2),
            encoding="utf-8",
        )

        self._append_task_log(
            task_dir,
            "stdout",
            (
                f"[upload] prepared scene with context={context_indices} "
                f"and target range {target_indices[0]}..{target_indices[-1]}"
            ),
        )
        self.logger.info(
            "uploaded media materialized with estimated cameras",
            event="uploaded_media_materialized",
            fields={
                "scene_key": scene_key,
                "preset": preset.name,
                "frame_count": len(prepared_paths),
                "camera_backend": camera_backend,
                "context_indices": context_indices,
                "target_start": target_indices[0],
                "target_end": target_indices[-1],
                "has_video": video_path is not None,
            },
        )

        frames_camera_data: list[dict] = []
        for i in range(len(prediction.extrinsics_w2c)):
            extrinsic = prediction.extrinsics_w2c[i]
            intrinsic = prediction.intrinsics_px[i]
            frames_camera_data.append({
                "index": i,
                "fx": float(intrinsic[0, 0]),
                "fy": float(intrinsic[1, 1]),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2]),
                "rotation": extrinsic[:, :3].tolist(),
                "translation": extrinsic[:, 3].tolist(),
            })
        self._save_camera_params(task_dir, frames_camera_data, source_width, source_height)

        return {
            "scene_key": scene_key,
            "dataset_root": dataset_root.parent,
            "evaluation_index_path": eval_meta_path,
            "images_dir": manual_images_dir,
            "poses_path": manual_poses_path,
            "target_poses_path": manual_target_poses_path,
            "input_preview_files": preview_files,
            "context_indices": context_indices,
            "target_indices": target_indices,
            "target_count": len(target_indices),
        }

    def materialize_images_with_poses(
        self,
        task_dir: Path,
        image_paths: list[str],
        pose_file_path: str,
        scene_key: str,
        preset_name: str,
        requested_context_indices: list[int],
    ) -> dict:
        preset = self._preset(preset_name)
        source_height, source_width = self._source_image_shape(preset)

        load_poses, _ = _pose_loader()
        pose_data = load_poses(pose_path=Path(pose_file_path), image_paths=[Path(p) for p in image_paths])

        if len(pose_data.image_paths) < preset.expected_min_views:
            raise ValueError(
                f"Preset {preset.display_name} requires at least {preset.expected_min_views} views, "
                f"pose file has {len(pose_data.image_paths)}"
            )

        context_indices = self._resolve_context_indices(
            requested_context_indices, len(pose_data.image_paths), preset.num_context_views
        )
        target_indices = list(range(min(context_indices), max(context_indices) + 1))

        dataset_root = task_dir / "input" / "dataset" / "test"
        dataset_root.mkdir(parents=True, exist_ok=True)
        prepared_dir = task_dir / "input" / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = task_dir / "input" / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        prepared_paths: list[Path] = []
        encoded_images: list[torch.Tensor] = []
        preview_files: list[str] = []
        context_index_set = set(context_indices)

        for order, img_path in enumerate(pose_data.image_paths):
            with Image.open(img_path) as image:
                prepared = self._prepare_uploaded_image(image, (source_height, source_width))
                filename = f"frame_{order:04d}.png"
                prepared_path = prepared_dir / filename
                prepared.save(prepared_path)
                prepared_paths.append(prepared_path)

                if order in context_index_set:
                    preview_path = preview_dir / filename
                    prepared.save(preview_path)
                    preview_files.append(f"input/preview/{filename}")

                encoded_images.append(self._encode_image(prepared))

        cameras = self._serialize_cameras(
            pose_data.extrinsics.numpy(),
            pose_data.intrinsics.numpy(),
            source_width,
            source_height,
        )

        example = {
            "url": "upload",
            "timestamps": torch.arange(len(prepared_paths), dtype=torch.int64),
            "cameras": torch.tensor(np.asarray(cameras), dtype=torch.float32),
            "images": encoded_images,
            "key": scene_key,
        }
        torch.save([example], dataset_root / "000000.torch")
        (dataset_root / "index.json").write_text(
            json.dumps({scene_key: "000000.torch"}, indent=2), encoding="utf-8"
        )

        eval_meta_path = task_dir / "meta" / "evaluation_index.json"
        eval_meta_path.write_text(
            json.dumps({scene_key: {"context": context_indices, "target": target_indices}}, indent=2),
            encoding="utf-8",
        )

        self._append_task_log(
            task_dir,
            "stdout",
            f"[upload+poses] prepared {len(prepared_paths)} frames with user-provided cameras",
        )

        extrinsics_np = pose_data.extrinsics.numpy() if hasattr(pose_data.extrinsics, "numpy") else np.array(pose_data.extrinsics)
        intrinsics_np = pose_data.intrinsics.numpy() if hasattr(pose_data.intrinsics, "numpy") else np.array(pose_data.intrinsics)
        frames_camera_data: list[dict] = []
        for i in range(len(extrinsics_np)):
            extrinsic = extrinsics_np[i]
            intrinsic = intrinsics_np[i]
            frames_camera_data.append({
                "index": i,
                "fx": float(intrinsic[0, 0]),
                "fy": float(intrinsic[1, 1]),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2]),
                "rotation": extrinsic[:, :3].tolist(),
                "translation": extrinsic[:, 3].tolist(),
            })
        self._save_camera_params(task_dir, frames_camera_data, source_width, source_height)

        return {
            "scene_key": scene_key,
            "dataset_root": dataset_root.parent,
            "evaluation_index_path": eval_meta_path,
            "image_paths": [str(p) for p in prepared_paths],
            "extrinsics": pose_data.extrinsics,
            "intrinsics": pose_data.intrinsics,
            "input_preview_files": preview_files,
            "context_indices": context_indices,
            "target_indices": target_indices,
            "target_count": len(target_indices),
        }

    def materialize_images_with_camera_params(
        self,
        task_dir: Path,
        image_paths: list[str],
        camera_params_path: str,
        scene_key: str,
        preset_name: str,
        requested_context_indices: list[int],
    ) -> dict:
        preset = self._preset(preset_name)
        source_height, source_width = self._source_image_shape(preset)
        resolved_image_paths = [Path(path) for path in image_paths if Path(path).exists()]
        if len(resolved_image_paths) < preset.expected_min_views:
            raise ValueError(
                f"Preset {preset.display_name} requires at least {preset.expected_min_views} views, "
                f"got {len(resolved_image_paths)} images"
            )

        camera_items = self._load_camera_params(camera_params_path)
        camera_by_index = {int(item["index"]): item for item in camera_items}
        available_indices = sorted(camera_by_index)
        if len(available_indices) < len(resolved_image_paths):
            raise ValueError(
                "camera_params.json has fewer camera entries than uploaded images: "
                f"{len(available_indices)} < {len(resolved_image_paths)}"
            )

        filename_indices = self._parse_frame_indices_from_paths(resolved_image_paths)
        if requested_context_indices:
            context_indices = self._resolve_context_indices(
                requested_context_indices, len(available_indices), preset.num_context_views
            )
        elif filename_indices is not None:
            context_indices = filename_indices
        else:
            context_indices = available_indices[: len(resolved_image_paths)]
            if len(context_indices) != preset.num_context_views:
                raise ValueError(
                    f"Preset requires exactly {preset.num_context_views} context images, "
                    f"got {len(resolved_image_paths)}"
                )

        missing = [index for index in context_indices if index not in camera_by_index]
        if missing:
            raise ValueError(f"camera_params.json is missing camera entries for indices: {missing}")
        if len(resolved_image_paths) != len(context_indices):
            raise ValueError(
                "Uploaded image count must match selected context indices: "
                f"{len(resolved_image_paths)} != {len(context_indices)}"
            )

        dataset_root = task_dir / "input" / "dataset" / "test"
        dataset_root.mkdir(parents=True, exist_ok=True)
        preview_dir = task_dir / "input" / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        encoded_images: list[torch.Tensor] = []
        preview_files: list[str] = []
        cameras: list[list[float]] = []
        frames_camera_data: list[dict] = []

        for order, (image_path, source_index) in enumerate(zip(resolved_image_paths, context_indices)):
            with Image.open(image_path) as image:
                prepared = self._prepare_uploaded_image(image, (source_height, source_width))
                encoded_images.append(self._encode_image(prepared))
                preview_name = f"context_{order:02d}_{source_index:06d}.png"
                prepared.save(preview_dir / preview_name)
                preview_files.append(f"input/preview/{preview_name}")

            cam = camera_by_index[source_index]
            cameras.append(self._camera_params_item_to_18d(cam))
            frames_camera_data.append({
                "index": order,
                "fx": float(cam["fx"]),
                "fy": float(cam["fy"]),
                "cx": float(cam["cx"]),
                "cy": float(cam["cy"]),
                "rotation": cam["rotation"],
                "translation": cam["translation"],
            })

        replay_indices = list(range(len(encoded_images)))
        example = {
            "url": "camera_params_replay",
            "timestamps": torch.arange(len(encoded_images), dtype=torch.int64),
            "cameras": torch.tensor(np.asarray(cameras), dtype=torch.float32),
            "images": encoded_images,
            "key": scene_key,
        }
        torch.save([example], dataset_root / "000000.torch")
        (dataset_root / "index.json").write_text(
            json.dumps({scene_key: "000000.torch"}, indent=2), encoding="utf-8"
        )

        eval_meta_path = task_dir / "meta" / "evaluation_index.json"
        eval_meta_path.parent.mkdir(parents=True, exist_ok=True)
        eval_meta_path.write_text(
            json.dumps({scene_key: {"context": replay_indices, "target": replay_indices}}, indent=2),
            encoding="utf-8",
        )
        self._save_camera_params(task_dir, frames_camera_data, source_width, source_height)
        self._append_task_log(
            task_dir,
            "stdout",
            (
                "[replay] prepared context-only dataset from uploaded images "
                f"and camera_params.json with source_indices={context_indices}"
            ),
        )

        return {
            "scene_key": scene_key,
            "dataset_root": dataset_root.parent,
            "evaluation_index_path": eval_meta_path,
            "input_preview_files": preview_files,
            "context_indices": replay_indices,
            "target_indices": replay_indices,
            "target_count": len(replay_indices),
            "use_script": True,
        }

    def _resolve_eval_indices(self, preset: PresetConfig, scene_key: str) -> tuple[list[int], list[int]]:
        eval_index = json.loads(preset.fixed_index_path.read_text(encoding="utf-8"))
        entry = eval_index.get(scene_key)
        if not entry:
            raise ValueError(f"No evaluation entry found for {scene_key}")
        context = list(entry["context"])
        target = list(entry["target"])
        return context, target

    def _preset(self, preset_name: str) -> PresetConfig:
        if preset_name not in self.settings.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        return self.settings.presets[preset_name]

    def _resolve_source_paths(
        self,
        task_dir: Path,
        work_dir: Path,
        image_paths: list[str],
        video_path: str | None,
        preset: PresetConfig,
        requested_context_indices: list[int],
    ) -> list[Path]:
        if video_path is not None:
            max_frames = None if requested_context_indices else preset.max_auto_video_frames
            if self._should_use_ttt3r_video_strategy(preset, video_path, requested_context_indices):
                return self._extract_video_frames_ttt3r_contiguous(
                    task_dir,
                    work_dir,
                    Path(video_path),
                    max_frames,
                    self._source_image_shape(preset),
                    preset.name,
                )
            return self._extract_video_frames(task_dir, work_dir, Path(video_path), max_frames)
        paths = [Path(path) for path in image_paths if Path(path).exists()]
        if not paths:
            raise ValueError("No uploaded images were found")
        return paths

    @staticmethod
    def _should_use_ttt3r_video_strategy(
        preset: PresetConfig,
        video_path: str | None,
        requested_context_indices: list[int],
    ) -> bool:
        return (
            video_path is not None
            and not requested_context_indices
            and preset.name in {
                "re10k_large_2view",
                "re10k_base_2view",
                "re10k_small_2view",
                "re10k_large_4view",
                "re10k_base_4view",
                "re10k_small_4view",
                "re10k_large_6view",
                "re10k_base_6view",
                "re10k_small_6view",
            }
        )

    def _camera_backend_for_upload(
        self,
        preset: PresetConfig,
        video_path: str | None,
        requested_context_indices: list[int],
    ) -> str:
        # Manual/uploaded scenes use the configured camera backend. The default
        # in settings is TTT3R at /root/TTT3R.
        return self.settings.camera_backend

    def _extract_video_frames_ttt3r_contiguous(
        self,
        task_dir: Path,
        work_dir: Path,
        video_path: Path,
        max_frames: int | None,
        target_shape: tuple[int, int],
        preset_name: str,
    ) -> list[Path]:
        if max_frames is None or max_frames <= 0:
            return self._extract_video_frames(task_dir, work_dir, video_path, max_frames)

        total_frames = self._probe_video_frame_count(video_path)
        if total_frames is None:
            self._append_task_log(
                task_dir,
                "stdout",
                f"[video] TTT3R contiguous-clip strategy for {preset_name} could not determine frame count; falling back to sparse capped sampling",
            )
            return self._extract_video_frames(task_dir, work_dir, video_path, max_frames)

        if total_frames <= max_frames:
            self._append_task_log(
                task_dir,
                "stdout",
                (
                    f"[video] TTT3R contiguous-clip strategy for {preset_name} kept the full clip because "
                    f"{total_frames} <= cap {max_frames}"
                ),
            )
            frame_paths = self._extract_video_frame_range(
                task_dir,
                work_dir / "video_frames",
                video_path,
                0,
                total_frames - 1,
                f"[video] extracting contiguous clip from {video_path.name}",
                target_shape=target_shape,
            )
            self._append_task_log(task_dir, "stdout", f"[video] extracted {len(frame_paths)} frames")
            return frame_paths

        probe_count = min(total_frames, max(max_frames, 24))
        probe_indices = self._sample_frame_indices_evenly(total_frames, probe_count)
        self._append_task_log(
            task_dir,
            "stdout",
            (
                "[video] probing "
                f"{len(probe_indices)} / {total_frames} frames to estimate the valid contiguous clip"
            ),
        )
        probe_paths = self._extract_video_frame_indices(
            task_dir,
            work_dir / "video_probe_frames",
            video_path,
            probe_indices,
            f"[video] extracting probe frames from {video_path.name}",
        )
        probe_start, probe_end = self._find_dark_edge_bounds(probe_paths)
        if probe_start == probe_end == -1:
            probe_start, probe_end = 0, len(probe_paths) - 1
        if probe_end - probe_start + 1 < 6:
            self._append_task_log(
                task_dir,
                "stdout",
                "[video] probe dark-frame trimming left too few samples; using the full probed range instead",
            )
            probe_start, probe_end = 0, len(probe_paths) - 1
        elif probe_start != 0 or probe_end != len(probe_paths) - 1:
            self._append_task_log(
                task_dir,
                "stdout",
                (
                    "[video] trimmed dark probe frames: dropped "
                    f"{probe_start} leading and {len(probe_paths) - probe_end - 1} trailing sampled frames"
                ),
            )

        valid_start = probe_indices[probe_start]
        valid_end = probe_indices[probe_end]
        window_start, window_end = self._select_middle_frame_window(valid_start, valid_end, max_frames)
        self._append_task_log(
            task_dir,
                "stdout",
                (
                f"[video] selected contiguous TTT3R clip for {preset_name} "
                f"{window_start}..{window_end} ({window_end - window_start + 1} frames) "
                f"from valid range {valid_start}..{valid_end}"
                ),
        )

        frame_paths = self._extract_video_frame_range(
            task_dir,
            work_dir / "video_frames",
            video_path,
            window_start,
            window_end,
            f"[video] extracting contiguous clip from {video_path.name}",
            target_shape=target_shape,
        )
        self._append_task_log(task_dir, "stdout", f"[video] extracted {len(frame_paths)} frames")
        return frame_paths

    def _extract_video_frames(
        self,
        task_dir: Path,
        work_dir: Path,
        video_path: Path,
        max_frames: int | None,
    ) -> list[Path]:
        if not video_path.exists():
            raise ValueError(f"Uploaded video not found: {video_path}")
        frames_dir = work_dir / "video_frames"
        select_expr = None
        total_frames = self._probe_video_frame_count(video_path)
        if max_frames is not None and total_frames is not None and total_frames > max_frames:
            selected_indices = self._sample_frame_indices_evenly(total_frames, max_frames)
            select_terms = [f"eq(n\\,{index})" for index in selected_indices]
            select_expr = "+".join(select_terms)
            self._append_task_log(
                task_dir,
                "stdout",
                f"[video] sampling {len(selected_indices)} / {total_frames} frames for preset cap {max_frames}",
            )
        frame_paths = self._extract_video_frames_with_select(
            task_dir,
            frames_dir,
            video_path,
            select_expr,
            f"[video] extracting frames from {video_path.name}",
        )
        if max_frames is not None and len(frame_paths) > max_frames:
            # ffmpeg may still overshoot by one or two frames depending on container timing.
            frame_paths = self._trim_extracted_frames(frame_paths, max_frames)
        self._append_task_log(task_dir, "stdout", f"[video] extracted {len(frame_paths)} frames")
        return frame_paths

    def _extract_video_frames_with_select(
        self,
        task_dir: Path,
        frames_dir: Path,
        video_path: Path,
        select_expr: str | None,
        extraction_message: str,
        target_shape: tuple[int, int] | None = None,
    ) -> list[Path]:
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = frames_dir / "frame_%06d.png"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
        ]
        filters = []
        if select_expr is not None:
            filters.append(f"select='{select_expr}'")
        if target_shape is not None:
            target_height, target_width = target_shape
            filters.append(
                (
                    f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                    f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2"
                )
            )
        if filters:
            cmd.extend(["-vf", ",".join(filters), "-vsync", "0"])
        cmd.append(str(output_pattern))
        self._append_task_log(task_dir, "stdout", extraction_message)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.stdout:
            self._append_task_log(task_dir, "stdout", result.stdout.rstrip())
        if result.stderr:
            self._append_task_log(task_dir, "stderr", result.stderr.rstrip())
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed while extracting frames: {result.stderr.strip() or result.returncode}")

        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        if not frame_paths:
            raise ValueError("No frames were extracted from the uploaded video")
        return frame_paths

    def _extract_video_frame_indices(
        self,
        task_dir: Path,
        frames_dir: Path,
        video_path: Path,
        frame_indices: list[int],
        extraction_message: str,
        target_shape: tuple[int, int] | None = None,
    ) -> list[Path]:
        select_terms = [f"eq(n\\,{index})" for index in frame_indices]
        return self._extract_video_frames_with_select(
            task_dir,
            frames_dir,
            video_path,
            "+".join(select_terms),
            extraction_message,
            target_shape=target_shape,
        )

    def _extract_video_frame_range(
        self,
        task_dir: Path,
        frames_dir: Path,
        video_path: Path,
        start_index: int,
        end_index: int,
        extraction_message: str,
        target_shape: tuple[int, int] | None = None,
    ) -> list[Path]:
        return self._extract_video_frames_with_select(
            task_dir,
            frames_dir,
            video_path,
            f"between(n\\,{start_index}\\,{end_index})",
            extraction_message,
            target_shape=target_shape,
        )

    def _trim_dark_video_edges(
        self,
        source_paths: list[Path],
        preset: PresetConfig,
        requested_context_indices: list[int],
        video_path: str | None,
        task_dir: Path,
    ) -> list[Path]:
        if video_path is None or requested_context_indices or len(source_paths) <= preset.expected_min_views:
            return source_paths

        start, end = self._find_dark_edge_bounds(source_paths)

        if start == 0 and end == len(source_paths) - 1:
            return source_paths
        if start == end == -1:
            return source_paths
        if end - start + 1 < preset.expected_min_views:
            self._append_task_log(
                task_dir,
                "stdout",
                "[video] detected dark edge frames but kept the full range because trimming would leave too few frames",
            )
            return source_paths

        trimmed = source_paths[start : end + 1]
        self._append_task_log(
            task_dir,
            "stdout",
            f"[video] trimmed dark edge frames: dropped {start} leading and {len(source_paths) - end - 1} trailing frames",
        )
        return trimmed

    def _find_dark_edge_bounds(self, frame_paths: list[Path]) -> tuple[int, int]:
        start = 0
        end = len(frame_paths) - 1
        while start < len(frame_paths) and self._is_dark_frame(frame_paths[start]):
            start += 1
        while end >= start and self._is_dark_frame(frame_paths[end]):
            end -= 1
        if start > end:
            return -1, -1
        return start, end

    def _probe_video_frame_count(self, video_path: Path) -> int | None:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0:
            return None
        raw = result.stdout.strip()
        if not raw or raw == "N/A":
            return None
        try:
            count = int(raw)
        except ValueError:
            return None
        return count if count > 0 else None

    @staticmethod
    def _resolve_context_indices(requested_context_indices: list[int], frame_count: int, required_count: int) -> list[int]:
        if requested_context_indices:
            context_indices = sorted(set(int(index) for index in requested_context_indices))
            if len(context_indices) != required_count:
                raise ValueError(f"Preset requires exactly {required_count} context indices, got {len(context_indices)}")
            if context_indices[-1] >= frame_count:
                raise ValueError(f"Context index {context_indices[-1]} is out of range for {frame_count} frames")
            return context_indices
        return SampleService._select_context_indices(frame_count, required_count)

    @staticmethod
    def _prepare_uploaded_image(image: Image.Image, target_shape: tuple[int, int]) -> Image.Image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
        target_height, target_width = target_shape
        scale = min(target_width / image.width, target_height / image.height)
        resized_width = max(1, round(image.width * scale))
        resized_height = max(1, round(image.height * scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        offset_x = (target_width - resized_width) // 2
        offset_y = (target_height - resized_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    @staticmethod
    def _encode_image(image: Image.Image) -> torch.Tensor:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()
        return torch.from_numpy(data)

    @staticmethod
    def _encode_image_file(image_path: Path) -> torch.Tensor:
        data = np.frombuffer(image_path.read_bytes(), dtype=np.uint8).copy()
        return torch.from_numpy(data)

    @staticmethod
    def _write_manual_inputs(
        task_dir: Path,
        prepared_paths: list[Path],
        extrinsics_w2c: np.ndarray,
        intrinsics_px: np.ndarray,
        width: int,
        height: int,
        context_indices: list[int] | None = None,
        target_indices: list[int] | None = None,
    ) -> tuple[Path, Path, Path | None]:
        manual_dir = task_dir / "input" / "manual"
        images_dir = manual_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        w2c_4x4 = []
        for extrinsic in extrinsics_w2c:
            sanitized = SampleService._sanitize_extrinsic_for_depthsplat(extrinsic)
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :4] = sanitized
            w2c_4x4.append(matrix)
        c2w = np.linalg.inv(np.stack(w2c_4x4)).astype(np.float32)

        intrinsics = np.asarray(intrinsics_px, dtype=np.float32).copy()
        intrinsics[:, 0, 0] /= float(width)
        intrinsics[:, 0, 2] /= float(width)
        intrinsics[:, 1, 1] /= float(height)
        intrinsics[:, 1, 2] /= float(height)

        selected_context = list(context_indices) if context_indices is not None else list(range(len(prepared_paths)))
        for index in selected_context:
            if index < 0 or index >= len(prepared_paths):
                raise ValueError(f"context index out of range for manual inputs: {index}")

        stored_paths: list[Path] = []
        for output_index, source_index in enumerate(selected_context):
            dst = images_dir / f"frame_{output_index:04d}.png"
            shutil.copy2(prepared_paths[source_index], dst)
            stored_paths.append(dst)

        poses_path = manual_dir / "poses.npz"
        np.savez(
            poses_path,
            extrinsics=c2w[selected_context],
            intrinsics=intrinsics[selected_context],
            image_paths=np.asarray([str(path) for path in stored_paths]),
            h=np.asarray(height, dtype=np.int32),
            w=np.asarray(width, dtype=np.int32),
        )

        target_poses_path: Path | None = None
        if target_indices is not None:
            selected_target = list(target_indices)
            for index in selected_target:
                if index < 0 or index >= len(prepared_paths):
                    raise ValueError(f"target index out of range for manual inputs: {index}")
            target_poses_path = manual_dir / "target_poses.npz"
            np.savez(
                target_poses_path,
                extrinsics=c2w[selected_target],
                intrinsics=intrinsics[selected_target],
                h=np.asarray(height, dtype=np.int32),
                w=np.asarray(width, dtype=np.int32),
            )

        return images_dir, poses_path, target_poses_path

    @staticmethod
    def _serialize_cameras(
        extrinsics_w2c: np.ndarray,
        intrinsics_px: np.ndarray,
        width: int,
        height: int,
    ) -> list[list[float]]:
        cameras: list[list[float]] = []
        for extrinsic, intrinsic in zip(extrinsics_w2c, intrinsics_px):
            sanitized_extrinsic = SampleService._sanitize_extrinsic_for_depthsplat(extrinsic)
            camera = [
                float(intrinsic[0, 0] / width),
                float(intrinsic[1, 1] / height),
                float(intrinsic[0, 2] / width),
                float(intrinsic[1, 2] / height),
                0.0,
                0.0,
            ]
            camera.extend(float(value) for value in sanitized_extrinsic.reshape(-1))
            cameras.append(camera)
        return cameras

    @staticmethod
    def _load_camera_params(camera_params_path: str) -> list[dict]:
        path = Path(camera_params_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        images = data.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("camera_params.json must contain a non-empty images array")

        seen: set[int] = set()
        required = {"index", "width", "height", "fx", "fy", "cx", "cy", "rotation", "translation"}
        for item in images:
            if not isinstance(item, dict):
                raise ValueError("camera_params.json images entries must be objects")
            missing = sorted(required - set(item))
            if missing:
                raise ValueError(f"camera_params.json image entry is missing fields: {missing}")
            index = int(item["index"])
            if index in seen:
                raise ValueError(f"camera_params.json has duplicate camera index: {index}")
            seen.add(index)
            rotation = np.asarray(item["rotation"], dtype=np.float32)
            translation = np.asarray(item["translation"], dtype=np.float32)
            if rotation.shape != (3, 3):
                raise ValueError(f"camera_params.json rotation for index {index} must be 3x3")
            if translation.shape != (3,):
                raise ValueError(f"camera_params.json translation for index {index} must have length 3")

        return sorted(images, key=lambda item: int(item["index"]))

    @staticmethod
    def _camera_params_item_to_18d(item: dict) -> list[float]:
        width = float(item["width"])
        height = float(item["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid camera dimensions for index {item['index']}: {width}x{height}")
        extrinsic = np.concatenate(
            [
                np.asarray(item["rotation"], dtype=np.float32),
                np.asarray(item["translation"], dtype=np.float32).reshape(3, 1),
            ],
            axis=1,
        )
        sanitized_extrinsic = SampleService._sanitize_extrinsic_for_depthsplat(extrinsic)
        camera = [
            float(item["fx"]) / width,
            float(item["fy"]) / height,
            float(item["cx"]) / width,
            float(item["cy"]) / height,
            0.0,
            0.0,
        ]
        camera.extend(float(value) for value in sanitized_extrinsic.reshape(-1))
        return camera

    @staticmethod
    def _sanitize_extrinsic_for_depthsplat(extrinsic_w2c: np.ndarray) -> np.ndarray:
        sanitized = np.asarray(extrinsic_w2c, dtype=np.float32)
        if sanitized.shape == (4, 4):
            sanitized = sanitized[:3, :4]
        elif sanitized.shape != (3, 4):
            raise ValueError(f"Expected a 3x4 or 4x4 extrinsic matrix, got {sanitized.shape}")
        sanitized = sanitized.copy()
        rotation = sanitized[:, :3]
        translation = sanitized[:, 3]

        # DepthSplat's SH rotation path assumes a true SO(3) rotation with det(R)=1.
        u, _, vh = np.linalg.svd(rotation)
        rotation_ortho = u @ vh
        if np.linalg.det(rotation_ortho) < 0:
            u[:, -1] *= -1.0
            rotation_ortho = u @ vh

        sanitized[:, :3] = rotation_ortho.astype(np.float32)
        sanitized[:, 3] = translation.astype(np.float32)
        return sanitized

    @staticmethod
    def _deserialize_cameras_from_18d(
        cameras_18d: np.ndarray,
        source_width: int,
        source_height: int,
    ) -> list[dict]:
        result: list[dict] = []
        for i in range(cameras_18d.shape[0]):
            cam_vec = cameras_18d[i]
            fx = float(cam_vec[0] * source_width)
            fy = float(cam_vec[1] * source_height)
            cx = float(cam_vec[2] * source_width)
            cy = float(cam_vec[3] * source_height)
            rotation = [[float(cam_vec[6]), float(cam_vec[7]), float(cam_vec[8])],
                        [float(cam_vec[9]), float(cam_vec[10]), float(cam_vec[11])],
                        [float(cam_vec[12]), float(cam_vec[13]), float(cam_vec[14])]]
            translation = [float(cam_vec[15]), float(cam_vec[16]), float(cam_vec[17])]
            result.append({
                "index": i,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "rotation": rotation,
                "translation": translation,
            })
        return result

    @staticmethod
    def _save_camera_params(task_dir: Path, frames_camera_data: list[dict], image_width: int, image_height: int) -> None:
        meta_dir = task_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        images: list[dict] = []
        for item in frames_camera_data:
            images.append({
                "index": item["index"],
                "width": image_width,
                "height": image_height,
                "fx": item["fx"],
                "fy": item["fy"],
                "cx": item["cx"],
                "cy": item["cy"],
                "rotation": item["rotation"],
                "translation": item["translation"],
            })
        (meta_dir / "camera_params.json").write_text(
            json.dumps({"images": images}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _sample_frame_paths_evenly(frame_paths: list[Path], max_frames: int) -> list[Path]:
        if len(frame_paths) <= max_frames:
            return frame_paths
        indices = np.round(np.linspace(0, len(frame_paths) - 1, max_frames)).astype(int).tolist()
        selected: list[Path] = []
        seen: set[int] = set()
        for index in indices:
            if index in seen:
                continue
            seen.add(index)
            selected.append(frame_paths[index])
        if len(selected) < max_frames:
            for index, path in enumerate(frame_paths):
                if index in seen:
                    continue
                selected.append(path)
                seen.add(index)
                if len(selected) == max_frames:
                    break
        return selected

    @staticmethod
    def _parse_frame_indices_from_paths(paths: list[Path]) -> list[int] | None:
        indices: list[int] = []
        for path in paths:
            match = None
            for candidate in re.finditer(r"(\d{6,}|\d+)", path.stem):
                match = candidate
            if match is None:
                return None
            indices.append(int(match.group(1)))
        if len(set(indices)) != len(indices):
            return None
        return indices

    @staticmethod
    def _is_dark_frame(frame_path: Path, mean_threshold: float = 5.0, max_threshold: float = 24.0) -> bool:
        with Image.open(frame_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return float(pixels.mean()) <= mean_threshold and float(pixels.max()) <= max_threshold

    @staticmethod
    def _sample_frame_indices_evenly(total_frames: int, max_frames: int) -> list[int]:
        if total_frames <= max_frames:
            return list(range(total_frames))
        indices = np.round(np.linspace(0, total_frames - 1, max_frames)).astype(int).tolist()
        selected: list[int] = []
        seen: set[int] = set()
        for index in indices:
            if index in seen:
                continue
            selected.append(index)
            seen.add(index)
        return selected

    @staticmethod
    def _select_middle_frame_window(valid_start: int, valid_end: int, max_frames: int) -> tuple[int, int]:
        if valid_end < valid_start:
            raise ValueError(f"Invalid frame window {valid_start}..{valid_end}")
        available = valid_end - valid_start + 1
        if available <= max_frames:
            return valid_start, valid_end
        offset = (available - max_frames) // 2
        start = valid_start + offset
        end = start + max_frames - 1
        return start, end

    def _trim_extracted_frames(self, frame_paths: list[Path], max_frames: int) -> list[Path]:
        kept = self._sample_frame_paths_evenly(frame_paths, max_frames)
        kept_set = {path.name for path in kept}
        for path in frame_paths:
            if path.name not in kept_set:
                path.unlink(missing_ok=True)
        return kept

    @staticmethod
    def _subsample_context_indices(context_indices: list[int], desired_count: int) -> list[int]:
        n = len(context_indices)
        if desired_count >= n:
            return context_indices
        selected_positions = np.round(np.linspace(0, n - 1, desired_count)).astype(int).tolist()
        return [context_indices[i] for i in selected_positions]

    @staticmethod
    def _select_context_indices(num_frames: int, context_count: int) -> list[int]:
        if num_frames < context_count:
            raise ValueError(f"Need at least {context_count} frames, got {num_frames}")
        if num_frames == context_count:
            return list(range(num_frames))

        rounded = np.round(np.linspace(0, num_frames - 1, context_count)).astype(int).tolist()
        deduped: list[int] = []
        seen: set[int] = set()
        for index in rounded:
            if index not in seen:
                deduped.append(index)
                seen.add(index)
        if len(deduped) < context_count:
            for index in range(num_frames):
                if index in seen:
                    continue
                deduped.append(index)
                seen.add(index)
                if len(deduped) == context_count:
                    break
        deduped.sort()
        return deduped

    @staticmethod
    def _select_context_indices_floor(num_frames: int, context_count: int) -> list[int]:
        if num_frames < context_count:
            raise ValueError(f"Need at least {context_count} frames, got {num_frames}")
        if num_frames == context_count:
            return list(range(num_frames))

        floor_spaced = np.linspace(0, num_frames - 1, context_count).astype(int).tolist()
        deduped: list[int] = []
        seen: set[int] = set()
        for index in floor_spaced:
            if index not in seen:
                deduped.append(index)
                seen.add(index)
        if len(deduped) < context_count:
            for index in range(num_frames):
                if index in seen:
                    continue
                deduped.append(index)
                seen.add(index)
                if len(deduped) == context_count:
                    break
        deduped.sort()
        return deduped

    @staticmethod
    def _source_image_shape(preset: PresetConfig) -> tuple[int, int]:
        shape = preset.source_image_shape or preset.ori_image_shape
        if shape is not None:
            return shape
        raise ValueError(f"Preset {preset.name} is missing source image shape configuration")

    def _append_task_log(self, task_dir: Path, stream: str, message: str) -> None:
        timestamp = now_local().isoformat()
        self.storage.append_text(task_dir / "logs" / f"{stream}.log", f"[{timestamp}] {message}")
