from __future__ import annotations

import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings
from app.models.enums import TaskState
from app.models.task import ResultMetadata, TaskRecord
from app.services.storage import FilesystemStorage
from app.utils.timezone import now_local

_ERROR_HEAT_SCALE = 64.0


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
        metrics, scores = self._collect_metrics(raw_output_dir)
        rendered_urls, gt_urls = self._copy_rendered_and_gt_images(task, raw_output_dir, task_dir)
        error_urls = self._build_error_overlay_images(task, task_dir)
        comparison_concat = self._build_comparison_concat(task, task_dir)
        images_zip_url = self._build_images_zip(task, raw_output_dir, task_dir)
        camera_params_url = self._expose_camera_params(task, task_dir)

        result = ResultMetadata(
            task_id=task.id,
            status=task.status,
            result_complete=bool(video_output or copied_depths),
            input_images=input_images,
            video_url=self.storage.artifact_url(task.id, video_output.name) if video_output else None,
            video_path=str(video_output) if video_output else None,
            depth_images=[self.storage.artifact_url(task.id, f"depth/{path.name}") for path in copied_depths],
            rendered_images=rendered_urls,
            gt_images=gt_urls,
            error_images=error_urls,
            comparison_concat_image=comparison_concat,
            images_zip_url=images_zip_url,
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
            scores=scores,
            error_summary=task.error_summary,
            cancel=task.cancel,
            splat_url=splat_url,
            splat_path=str(task_dir / f"{task.request.scene_key}.splat") if splat_url else None,
            conversion_error=conversion_error,
            camera_params_url=camera_params_url,
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

    def _scene_image_dir(self, raw_output_dir: Path, task: TaskRecord, kind: str) -> Path:
        candidates = [
            raw_output_dir / "images" / task.request.scene_key / kind,
            raw_output_dir / "images" / "scene" / kind,
            raw_output_dir / "metrics" / "images" / task.request.scene_key / kind,
            raw_output_dir / "metrics" / "images" / "scene" / kind,
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def _copy_depth_images(self, task: TaskRecord, raw_output_dir: Path, depth_dir: Path) -> list[Path]:
        raw_depth_root = self._scene_image_dir(raw_output_dir, task, "depth")
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

    def _copy_rendered_and_gt_images(
        self, task: TaskRecord, raw_output_dir: Path, task_dir: Path
    ) -> tuple[list[str], list[str]]:
        color_dir = self._scene_image_dir(raw_output_dir, task, "color")
        rendered_dir = task_dir / "rendered"
        gt_dir = task_dir / "gt"
        rendered_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        rendered_urls: list[str] = []
        gt_urls: list[str] = []
        stderr_log = self.storage.log_path(task.id, "stderr.log")

        if not color_dir.exists():
            return rendered_urls, gt_urls

        all_rendered = sorted(
            p for p in color_dir.glob("*.png")
            if not p.name.endswith("_gt.png") and not p.name.startswith("input_")
        )
        gt_by_frame = {
            path.stem.removesuffix("_gt"): path
            for path in color_dir.glob("*_gt.png")
        }

        n = len(all_rendered)
        if n == 0:
            return rendered_urls, gt_urls

        max_samples = 10
        if n <= max_samples:
            indices = list(range(n))
        else:
            indices = np.linspace(0, n - 1, max_samples, dtype=int).tolist()

        for idx in indices:
            rd_src = all_rendered[idx]
            gt_src = gt_by_frame.get(rd_src.stem)
            if gt_src is None:
                self._log(stderr_log, f"[result-images] missing ground truth for {rd_src.name}; skipped")
                continue

            rd_dst = rendered_dir / rd_src.name
            shutil.copy2(rd_src, rd_dst)
            rendered_urls.append(self.storage.artifact_url(task.id, f"rendered/{rd_src.name}"))

            gt_dst = gt_dir / gt_src.name
            shutil.copy2(gt_src, gt_dst)
            gt_urls.append(self.storage.artifact_url(task.id, f"gt/{gt_src.name}"))

        return rendered_urls, gt_urls

    def _build_comparison_concat(
        self, task: TaskRecord, task_dir: Path
    ) -> str | None:
        rendered_dir = task_dir / "rendered"
        gt_dir = task_dir / "gt"

        rendered_files = sorted(rendered_dir.glob("*.png"))
        gt_files = sorted(gt_dir.glob("*.png"))

        if not rendered_files or not gt_files:
            return None

        count = min(len(rendered_files), len(gt_files))
        rendered_files = rendered_files[:count]
        gt_files = gt_files[:count]

        rd_imgs = [Image.open(p) for p in rendered_files]
        gt_imgs = [Image.open(p) for p in gt_files]

        h = rd_imgs[0].height
        for img in rd_imgs + gt_imgs:
            if img.height != h:
                img = img.resize((int(img.width * h / img.height), h), Image.LANCZOS)

        total_w = sum(img.width for img in rd_imgs)
        canvas = Image.new("RGB", (total_w, h * 2))

        x_offset = 0
        for gt_img, rd_img in zip(gt_imgs, rd_imgs):
            canvas.paste(gt_img, (x_offset, 0))
            canvas.paste(rd_img, (x_offset, h))
            x_offset += rd_img.width

        filename = f"img_comparison_{task.request.scene_key}.png"
        output_path = task_dir / filename
        canvas.save(output_path, optimize=True)

        for img in rd_imgs + gt_imgs:
            img.close()

        return self.storage.artifact_url(task.id, filename)

    def _build_error_overlay_images(self, task: TaskRecord, task_dir: Path) -> list[str]:
        rendered_dir = task_dir / "rendered"
        gt_dir = task_dir / "gt"
        error_dir = task_dir / "error"
        error_dir.mkdir(parents=True, exist_ok=True)
        stderr_log = self.storage.log_path(task.id, "stderr.log")

        gt_by_frame = {
            path.stem.removesuffix("_gt"): path
            for path in gt_dir.glob("*_gt.png")
        }
        error_urls: list[str] = []
        for rendered_path in sorted(rendered_dir.glob("*.png")):
            gt_path = gt_by_frame.get(rendered_path.stem)
            if gt_path is None:
                self._log(stderr_log, f"[error-overlay] missing ground truth for {rendered_path.name}; skipped")
                continue

            output_name = f"{rendered_path.stem}_error.png"
            output_path = error_dir / output_name
            try:
                with Image.open(rendered_path) as rendered_image, Image.open(gt_path) as gt_image:
                    rendered_rgb = rendered_image.convert("RGB")
                    gt_rgb = gt_image.convert("RGB")
                    if rendered_rgb.size != gt_rgb.size:
                        self._log(
                            stderr_log,
                            (
                                f"[error-overlay] resizing {rendered_path.name} from "
                                f"{rendered_rgb.size} to {gt_rgb.size}"
                            ),
                        )
                        rendered_rgb = rendered_rgb.resize(gt_rgb.size, Image.Resampling.LANCZOS)

                    rendered = np.asarray(rendered_rgb, dtype=np.float32)
                    ground_truth = np.asarray(gt_rgb, dtype=np.float32)
                    base = (rendered + ground_truth) * 0.5
                    error = np.abs(rendered - ground_truth).mean(axis=2)
                    intensity = np.clip(error / _ERROR_HEAT_SCALE, 0.0, 1.0)[..., None]
                    red = np.zeros_like(base)
                    red[..., 0] = 255.0
                    overlay = np.clip(base * (1.0 - intensity) + red * intensity, 0.0, 255.0).astype(np.uint8)
                    Image.fromarray(overlay, mode="RGB").save(output_path, optimize=True)
            except Exception as exc:  # noqa: BLE001
                self._log(stderr_log, f"[error-overlay] failed for {rendered_path.name}: {exc}")
                continue

            error_urls.append(self.storage.artifact_url(task.id, f"error/{output_name}"))

        return error_urls

    def _build_images_zip(
        self, task: TaskRecord, raw_output_dir: Path, task_dir: Path
    ) -> str | None:
        color_dir = self._scene_image_dir(raw_output_dir, task, "color")
        if not color_dir.exists():
            return None

        files_to_zip = sorted(
            p for p in color_dir.glob("*.png")
            if not p.name.startswith("input_")
        )
        if not files_to_zip:
            return None

        zip_path = task_dir / "images.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files_to_zip:
                zf.write(f, f.name)

        return self.storage.artifact_url(task.id, "images.zip")

    def _expose_camera_params(self, task: TaskRecord, task_dir: Path) -> str | None:
        src = task_dir / "meta" / "camera_params.json"
        if not src.exists():
            return None
        dst = task_dir / "camera_params.json"
        shutil.copy2(src, dst)
        return self.storage.artifact_url(task.id, "camera_params.json")

    def _collect_metrics(self, raw_output_dir: Path) -> tuple[dict, dict]:
        metrics: dict = {}
        scores: dict = {}
        benchmark_path = raw_output_dir / "metrics" / "benchmark.json"
        peak_memory_path = raw_output_dir / "metrics" / "peak_memory.json"
        if benchmark_path.exists():
            metrics["benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if peak_memory_path.exists():
            metrics["peak_memory_bytes"] = json.loads(peak_memory_path.read_text(encoding="utf-8"))
        scores_path = raw_output_dir / "metrics" / "scores_per_scene.json"
        if not scores_path.exists():
            scores_path = raw_output_dir / "metrics" / "scores_all_avg.json"
        if scores_path.exists():
            try:
                raw_scores = json.loads(scores_path.read_text(encoding="utf-8"))
                if isinstance(raw_scores, list) and raw_scores:
                    raw_scores = raw_scores[0]
                for key in ("psnr", "ssim", "lpips", "mean_pmr"):
                    if key in raw_scores:
                        scores[key] = round(float(raw_scores[key]), 4)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        return metrics, scores

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
        timestamp = now_local().isoformat()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{timestamp}] {message}\n")
