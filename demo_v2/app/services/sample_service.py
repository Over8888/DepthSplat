from __future__ import annotations

import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from app.config import PresetConfig, Settings
from app.models.sample import SampleRecord
from app.utils.logging import get_logger


class SampleService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
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
        scene_keys = sorted(set(dataset_index).intersection(eval_index))[: preset.sample_limit]
        samples: list[SampleRecord] = []
        for key in scene_keys:
            samples.append(
                SampleRecord(
                    id=f"{preset.name}:{key}",
                    preset=preset.name,
                    scene_key=key,
                    label=f"{key} ({preset.display_name})",
                    source_chunk=(preset.dataset_root / preset.sample_stage / dataset_index[key]).resolve(),
                    defaults={
                        "checkpoint": preset.checkpoint_path.name,
                        "num_context_views": preset.num_context_views,
                        "image_shape": list(preset.image_shape),
                    },
                )
            )
        return samples

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

    def materialize_sample(self, task_dir: Path, sample_id: str, preset_name: str) -> dict:
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

        return {
            "scene_key": sample.scene_key,
            "dataset_root": dataset_root.parent,
            "evaluation_index_path": eval_meta_path,
            "input_preview_files": preview_files,
            "context_indices": context_indices,
            "target_count": len(target_indices),
        }

    def materialize_uploaded_images(self, task_dir: Path, image_paths: list[str], scene_key: str) -> dict:
        preview_dir = task_dir / "input" / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_files: list[str] = []
        for order, image_path in enumerate(image_paths):
            src = Path(image_path)
            if not src.exists():
                continue
            suffix = src.suffix.lower() or ".png"
            filename = f"upload_{order:02d}{suffix}"
            image = Image.open(src).convert("RGB")
            image.save(preview_dir / filename)
            preview_files.append(f"input/preview/{filename}")
        return {
            "scene_key": scene_key,
            "dataset_root": None,
            "evaluation_index_path": None,
            "input_preview_files": preview_files,
            "context_indices": list(range(len(preview_files))),
            "target_count": 0,
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
