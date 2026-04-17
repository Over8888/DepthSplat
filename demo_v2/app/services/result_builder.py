from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.models.enums import TaskState
from app.models.task import ResultMetadata, TaskRecord
from app.services.storage import FilesystemStorage


class ResultBuilder:
    def __init__(self, settings: Settings, storage: FilesystemStorage) -> None:
        self.settings = settings
        self.storage = storage

    def build(self, task: TaskRecord) -> tuple[ResultMetadata, dict]:
        task_dir = self.storage.task_dir(task.id)
        raw_output_dir = task_dir / "meta" / "depthsplat_output"
        depth_dir = task_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)

        video_output = self._copy_result_video(task.id, raw_output_dir, task_dir)
        copied_depths = self._copy_depth_images(task, raw_output_dir, depth_dir)
        preview_images, input_images = self._collect_input_images(task)
        splat_url, conversion_error, splat_conversion_seconds = self._build_splat(task, raw_output_dir, task_dir)
        metrics = self._collect_metrics(raw_output_dir)

        result = ResultMetadata(
            task_id=task.id,
            status=task.status,
            result_complete=bool(video_output or copied_depths),
            input_images=input_images,
            video_url=self.storage.artifact_url(task.id, video_output.name) if video_output else None,
            video_path=str(video_output) if video_output else None,
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
            splat_url=splat_url,
            splat_path=str(task_dir / f"{task.request.scene_key}.splat") if splat_url else None,
            conversion_error=conversion_error,
        )
        timing_patch = self._timing_patch(task, metrics, splat_conversion_seconds)
        return result, timing_patch

    def build_cancelled(self, task: TaskRecord) -> ResultMetadata:
        task_dir = self.storage.task_dir(task.id)
        preview_dir = task_dir / "input" / "preview"
        depth_dir = task_dir / "depth"
        result_video = task_dir / "result.mp4"
        return ResultMetadata(
            task_id=task.id,
            status=TaskState.cancelled,
            result_complete=False,
            input_images=[self.storage.artifact_url(task.id, f"input/preview/{path.name}") for path in sorted(preview_dir.glob("*.png"))],
            video_url=self.storage.artifact_url(task.id, result_video.name) if result_video.exists() else None,
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
            splat_url=None,
            splat_path=None,
            conversion_error=None,
        )

    def _copy_result_video(self, task_id: str, raw_output_dir: Path, task_dir: Path) -> Path | None:
        video_paths = sorted((raw_output_dir / "videos").glob("*.mp4"))
        if not video_paths:
            return None
        dst = task_dir / "result.mp4"
        shutil.copy2(video_paths[0], dst)
        return dst

    def _copy_depth_images(self, task: TaskRecord, raw_output_dir: Path, depth_dir: Path) -> list[Path]:
        raw_depth_root = raw_output_dir / "images" / task.request.scene_key / "depth"
        if not raw_depth_root.exists():
            raw_depth_root = raw_output_dir / "metrics" / "images" / task.request.scene_key / "depth"
        copied_depths: list[Path] = []
        if raw_depth_root.exists():
            for path in sorted(raw_depth_root.glob("*.png")):
                dst = depth_dir / path.name
                shutil.copy2(path, dst)
                copied_depths.append(dst)
        return copied_depths

    def _collect_input_images(self, task: TaskRecord) -> tuple[list[str], list[str]]:
        task_dir = self.storage.task_dir(task.id)
        preview_dir = task_dir / "input" / "preview"
        upload_dir = task_dir / "input"
        preview_images = [self.storage.artifact_url(task.id, f"input/preview/{path.name}") for path in sorted(preview_dir.glob("*.png"))]
        input_images = [self.storage.artifact_url(task.id, f"input/{path.name}") for path in sorted(upload_dir.glob("upload_*"))]
        if not input_images:
            input_images = list(preview_images)
        return preview_images, input_images

    def _build_splat(self, task: TaskRecord, raw_output_dir: Path, task_dir: Path) -> tuple[str | None, str | None, float | None]:
        scene_name = task.request.scene_key
        raw_ply_path = raw_output_dir / "gaussians" / f"{scene_name}.ply"
        stdout_log = self.storage.log_path(task.id, "stdout.log")
        stderr_log = self.storage.log_path(task.id, "stderr.log")

        if not raw_ply_path.exists():
            msg = f"Gaussian PLY file not found: {raw_ply_path}"
            self._log(stdout_log, f"[splat] {msg}")
            return None, msg, None

        ply_output = task_dir / f"{scene_name}.ply"
        splat_output = task_dir / f"{scene_name}.splat"
        shutil.copy2(raw_ply_path, ply_output)

        convert_script = self.settings.depthsplat_root / "convertSplat.py"
        cmd = [self.settings.depthsplat_python, str(convert_script), str(ply_output), "-o", str(splat_output)]
        self._log(stdout_log, f"[splat] starting convertSplat.py for {ply_output.name} -> {splat_output.name}")
        convert_started = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.settings.depthsplat_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"convertSplat.py execution failed: {exc}"
            self._log(stderr_log, f"[splat] {error}")
            return None, error, round(time.monotonic() - convert_started, 4)

        if result.stdout:
            self._log(stdout_log, result.stdout.rstrip())
        if result.stderr:
            self._log(stderr_log, result.stderr.rstrip())

        conversion_seconds = round(time.monotonic() - convert_started, 4)
        if result.returncode != 0:
            error = f"convertSplat.py exited with code {result.returncode}"
            self._log(stderr_log, f"[splat] {error}")
            return None, error, conversion_seconds
        if not splat_output.exists():
            error = f"convertSplat.py finished without output file: {splat_output}"
            self._log(stderr_log, f"[splat] {error}")
            return None, error, conversion_seconds

        self._log(stdout_log, f"[splat] generated {splat_output.name}")
        return self.storage.artifact_url(task.id, splat_output.name), None, conversion_seconds

    def _collect_metrics(self, raw_output_dir: Path) -> dict:
        metrics: dict = {}
        benchmark_path = raw_output_dir / "metrics" / "benchmark.json"
        peak_memory_path = raw_output_dir / "metrics" / "peak_memory.json"
        if benchmark_path.exists():
            metrics["benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if peak_memory_path.exists():
            metrics["peak_memory_bytes"] = json.loads(peak_memory_path.read_text(encoding="utf-8"))
        return metrics

    def _timing_patch(self, task: TaskRecord, metrics: dict, splat_conversion_seconds: float | None) -> dict:
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
            "splat_conversion_seconds": splat_conversion_seconds,
            "save_outputs_seconds": task.timing.save_outputs_seconds,
        }

    @staticmethod
    def _log(path: Path, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{timestamp}] {message}\n")
