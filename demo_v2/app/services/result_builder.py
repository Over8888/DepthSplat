from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path

from app.config import Settings
from app.models.enums import TaskState
from app.models.task import CancelMetadata, ResultMetadata, TaskRecord
from app.services.storage import FilesystemStorage


class ResultBuilder:
    def __init__(self, settings: Settings, storage: FilesystemStorage) -> None:
        self.settings = settings
        self.storage = storage

    def build(self, task: TaskRecord) -> tuple[ResultMetadata, dict]:
        task_dir = self.storage.task_dir(task.id)
        raw_output_dir = task_dir / "meta" / "depthsplat_output"
        render_dir = task_dir / "render"
        depth_dir = task_dir / "depth"
        render_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        video_paths = sorted((raw_output_dir / "videos").glob("*.mp4"))
        copied_videos = []
        for path in video_paths:
            dst = render_dir / path.name
            shutil.copy2(path, dst)
            copied_videos.append(dst)

        raw_depth_root = raw_output_dir / "images" / task.request.scene_key / "depth"
        if not raw_depth_root.exists():
            raw_depth_root = raw_output_dir / "metrics" / "images" / task.request.scene_key / "depth"
        copied_depths = []
        if raw_depth_root.exists():
            for path in sorted(raw_depth_root.glob("*.png")):
                dst = depth_dir / path.name
                shutil.copy2(path, dst)
                copied_depths.append(dst)

        preview_dir = task_dir / "input" / "preview"
        upload_dir = task_dir / "input"
        preview_images = [self.storage.artifact_url(task.id, f"input/preview/{path.name}") for path in sorted(preview_dir.glob("*.png"))]
        input_images = [self.storage.artifact_url(task.id, f"input/{path.name}") for path in sorted(upload_dir.glob("upload_*"))]
        if not input_images:
            input_images = list(preview_images)

        metrics = self._collect_metrics(raw_output_dir)
        result = ResultMetadata(
            task_id=task.id,
            status=task.status,
            result_complete=bool(copied_videos or copied_depths),
            input_images=input_images,
            video_url=self.storage.artifact_url(task.id, f"render/{copied_videos[0].name}") if copied_videos else None,
            video_path=str(copied_videos[0]) if copied_videos else None,
            depth_images=[self.storage.artifact_url(task.id, f"depth/{path.name}") for path in copied_depths],
            preview_images=preview_images,
            options=task.request.options,
            message="Task finished successfully",
            parameters={
                "preset": task.request.preset,
                "checkpoint": task.request.checkpoint,
                "image_shape": task.request.image_shape,
                "num_context_views": task.request.num_context_views,
            },
            metrics=metrics,
            error_summary=task.error_summary,
            cancel=task.cancel,
        )
        timing_patch = self._timing_patch(task, metrics)
        return result, timing_patch

    def build_cancelled(self, task: TaskRecord) -> ResultMetadata:
        task_dir = self.storage.task_dir(task.id)
        preview_dir = task_dir / "input" / "preview"
        depth_dir = task_dir / "depth"
        render_dir = task_dir / "render"
        video_paths = sorted(render_dir.glob("*.mp4"))
        return ResultMetadata(
            task_id=task.id,
            status=TaskState.cancelled,
            result_complete=False,
            input_images=[self.storage.artifact_url(task.id, f"input/preview/{path.name}") for path in sorted(preview_dir.glob("*.png"))],
            video_url=self.storage.artifact_url(task.id, f"render/{video_paths[0].name}") if video_paths else None,
            depth_images=[self.storage.artifact_url(task.id, f"depth/{path.name}") for path in sorted(depth_dir.glob("*.png"))],
            preview_images=[self.storage.artifact_url(task.id, f"input/preview/{path.name}") for path in sorted(preview_dir.glob("*.png"))],
            options=task.request.options,
            message="Task was cancelled",
            parameters={
                "preset": task.request.preset,
                "checkpoint": task.request.checkpoint,
                "image_shape": task.request.image_shape,
                "num_context_views": task.request.num_context_views,
            },
            metrics={},
            error_summary=task.error_summary,
            cancel=task.cancel,
        )

    def _collect_metrics(self, raw_output_dir: Path) -> dict:
        metrics: dict = {}
        benchmark_path = raw_output_dir / "metrics" / "benchmark.json"
        peak_memory_path = raw_output_dir / "metrics" / "peak_memory.json"
        if benchmark_path.exists():
            metrics["benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if peak_memory_path.exists():
            metrics["peak_memory_bytes"] = json.loads(peak_memory_path.read_text(encoding="utf-8"))
        return metrics

    def _timing_patch(self, task: TaskRecord, metrics: dict) -> dict:
        benchmark = metrics.get("benchmark", {})
        encoder = benchmark.get("encoder", [])
        decoder = benchmark.get("decoder", [])
        forward_seconds = None
        if encoder or decoder:
            forward_seconds = float(sum(encoder) + sum(decoder))
        model_load_seconds = None
        if task.timing.running_started_at and task.timing.postprocessing_started_at:
            runtime = (task.timing.postprocessing_started_at - task.timing.running_started_at).total_seconds()
            if forward_seconds is not None:
                model_load_seconds = max(runtime - forward_seconds, 0.0)
            else:
                model_load_seconds = runtime
        return {
            "forward_seconds": forward_seconds,
            "model_load_seconds": model_load_seconds,
            "save_outputs_seconds": task.timing.save_outputs_seconds,
        }
