from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock

from app.config import PresetConfig, Settings
from app.utils.logging import get_logger


@dataclass
class RunnerResult:
    return_code: int | None
    started_at: datetime
    finished_at: datetime
    runtime_seconds: float
    cancelled: bool = False


@dataclass
class RunningProcess:
    popen: subprocess.Popen
    task_id: str
    pgid: int | None
    cancel_event: Event


class Runner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self._lock = Lock()
        self._active: dict[str, RunningProcess] = {}

    def build_command(
        self,
        preset: PresetConfig,
        dataset_root: Path,
        eval_index_path: Path,
        output_dir: Path,
        options: dict | None = None,
    ) -> list[str]:
        options = options or {}
        save_video = bool(options.get("saveVideo", True))
        compute_scores = bool(options.get("computeScores", False))
        export_depth_map = bool(options.get("exportDepthMap", True))
        use_test_chunk_interval = bool(options.get("testChunkInterval", True))
        extra_overrides = list(preset.extra_overrides)
        if not use_test_chunk_interval:
            extra_overrides = [item for item in extra_overrides if not item.startswith("dataset.test_chunk_interval=")]
        cmd = [
            self.settings.depthsplat_python,
            "-m",
            "src.main",
            f"+experiment={preset.experiment}",
            f"dataset.roots=[{dataset_root}]",
            f"dataset.view_sampler.index_path={eval_index_path}",
            "dataset/view_sampler=evaluation",
            f"dataset.view_sampler.num_context_views={preset.num_context_views}",
            f"checkpointing.pretrained_model={preset.checkpoint_path}",
            "mode=test",
            f"test.save_video={'true' if save_video else 'false'}",
            f"test.save_depth={'true' if export_depth_map else 'false'}",
            f"test.save_depth_concat_img={'true' if export_depth_map else 'false'}",
            "test.save_gaussian=true",
            f"test.compute_scores={'true' if compute_scores else 'false'}",
            "test.save_image=false",
            "test.save_gt_image=false",
            "test.save_input_images=false",
            f"output_dir={output_dir}",
        ]
        cmd.extend(extra_overrides)
        return [str(part) for part in cmd]

    def build_script_env(
        self,
        script_path: str,
        dataset_root: Path | None,
        eval_index_path: Path | None,
        output_dir: Path,
        images_dir: Path | None,
        poses_path: Path | None,
        target_poses_path: Path | None = None,
        preset: PresetConfig | None = None,
        options: dict | None = None,
        scene_key: str | None = None,
    ) -> dict[str, str]:
        """构建脚本执行所需的环境变量"""
        options = options or {}
        env_vars = {
            "PROJECT_ROOT": str(self.settings.depthsplat_root),
            "PYTHON_BIN": str(self.settings.depthsplat_python),
            "OUTPUT_DIR": str(output_dir),
            "SKVIDEO_FFMPEG_PATH": "/usr/bin",
        }
        
        # sample 模式：提供数据集路径和评估索引
        if dataset_root is not None:
            env_vars["DATA_ROOT"] = str(dataset_root)
        if eval_index_path is not None:
            env_vars["EVAL_INDEX"] = str(eval_index_path)
        
        # manual 模式：提供图片目录和 pose 文件
        if images_dir is not None:
            env_vars["IMAGES"] = str(images_dir)
        if poses_path is not None:
            env_vars["POSES"] = str(poses_path)
        if target_poses_path is not None:
            env_vars["TARGET_POSES"] = str(target_poses_path)
        if scene_key is not None:
            env_vars["SCENE_KEY"] = scene_key

        if preset is not None:
            env_vars.update(self._build_manual_preset_env(preset))

        def _bool_env(name: str, *keys: str, default: bool | None = None) -> None:
            value = None
            for key in keys:
                if key in options:
                    value = bool(options.get(key))
                    break
            if value is None:
                if default is None:
                    return
                value = default
            env_vars[name] = "true" if value else "false"

        def _int_env(name: str, *keys: str, default: int | None = None) -> None:
            value = None
            for key in keys:
                if key in options and options.get(key) is not None:
                    value = int(options.get(key))
                    break
            if value is None:
                if default is None:
                    return
                value = default
            env_vars[name] = str(value)

        # 前端 options 映射为环境变量
        _bool_env("SAVE_VIDEO", "saveVideo", "save_video", default=True)
        _bool_env("COMPUTE_SCORES", "computeScores", "compute_scores", default=True)
        _bool_env("SAVE_IMAGE", "saveImage", "save_image", default=True)
        _bool_env("SAVE_GT_IMAGE", "saveGtImage", "save_gt_image", default=True)
        _bool_env("SAVE_INPUT_IMAGES", "saveInputImages", "save_input_images", default=True)
        _bool_env("SAVE_GAUSSIAN", "saveGaussian", "save_gaussian", default=True)
        _bool_env("SAVE_DEPTH", "exportDepthMap", "saveDepth", "save_depth", default=True)
        _bool_env("SAVE_DEPTH_CONCAT_IMG", "saveDepthConcatImg", "save_depth_concat_img", default=True)
        _bool_env("SAVE_DEPTH_NPY", "saveDepthNpy", "save_depth_npy", default=False)
        _bool_env("SAVE_PLY", "savePly", "save_ply", default=False)
        _bool_env("TEST_CHUNK_INTERVAL", "testChunkInterval", "test_chunk_interval", default=True)
        _int_env("RENDER_CHUNK_SIZE", "renderChunkSize", "render_chunk_size", default=10)
        _int_env("METRIC_CHUNK_SIZE", "metricChunkSize", "metric_chunk_size", default=4)

        return env_vars

    def _infer_manual_model_type(self, preset: PresetConfig) -> str:
        checkpoint_name = preset.checkpoint_path.name.lower()
        if "large" in checkpoint_name:
            return "vitl"
        if "small" in checkpoint_name:
            return "vits"
        return "vitb"

    def _build_manual_preset_env(self, preset: PresetConfig) -> dict[str, str]:
        is_re10k_2view = preset.num_context_views == 2 and preset.image_shape == (256, 256)
        vit_type = self._infer_manual_model_type(preset)
        env_vars = {
            "CKPT_PATH": str(preset.checkpoint_path),
            "VIT_TYPE": vit_type,
            "MAX_IMAGE_SIZE": str(max(preset.image_shape)),
            "IMAGE_SHAPE": f"{preset.image_shape[0]},{preset.image_shape[1]}",
            "ORI_IMAGE_SHAPE": (
                f"{preset.ori_image_shape[0]},{preset.ori_image_shape[1]}"
                if preset.ori_image_shape is not None
                else f"{preset.image_shape[0]},{preset.image_shape[1]}"
            ),
            "EXPERIMENT": preset.experiment,
            "NUM_CONTEXT_VIEWS": str(preset.num_context_views),
            "NUM_DEPTH_CANDIDATES": str(self.settings.demo_num_depth_candidates),
            "SH_DEGREE": str(self.settings.demo_sh_degree),
        }
        if is_re10k_2view:
            env_vars.update(
                {
                    "NUM_SCALES": "2" if vit_type != "vits" else "1",
                    "UPSAMPLE_FACTOR": "4" if vit_type == "vits" else "2",
                    "LOWEST_FEATURE_RESOLUTION": "4",
                    "GAUSSIAN_SCALE_MAX": "3.0",
                    "SHIM_PATCH_SIZE": str(self.settings.demo_shim_patch_size),
                }
            )
        else:
            env_vars.update(
                {
                    "NUM_SCALES": "2" if vit_type != "vits" else "1",
                    "UPSAMPLE_FACTOR": "8" if vit_type == "vits" else "4",
                    "LOWEST_FEATURE_RESOLUTION": "8",
                    "GAUSSIAN_SCALE_MAX": "0.1",
                    "SHIM_PATCH_SIZE": "16",
                }
            )
        return env_vars

    def write_command_files(self, task_dir: Path, preset: PresetConfig, command: list[str]) -> None:
        logs_dir = task_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        command_line = shlex.join(command)
        (logs_dir / "cmd.txt").write_text(
            "\n".join(
                [
                    f"workdir: {self.settings.depthsplat_root}",
                    f"preset: {preset.name}",
                    f"env: {json.dumps(preset.env, ensure_ascii=False)}",
                    f"command: {command_line}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (logs_dir / "cmd.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + f"cd {shlex.quote(str(self.settings.depthsplat_root))}\n"
            + "".join(f"export {key}={shlex.quote(value)}\n" for key, value in preset.env.items())
            + f"export SKVIDEO_FFMPEG_PATH={shlex.quote('/usr/bin')}\n"
            + command_line
            + "\n",
            encoding="utf-8",
        )
        os.chmod(logs_dir / "cmd.sh", 0o755)

    def write_script_command_files(
        self,
        task_dir: Path,
        script_path: str,
        env_vars: dict[str, str],
    ) -> None:
        """记录脚本执行命令到 cmd.txt 和 cmd.sh，便于排查"""
        logs_dir = task_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        env_lines = "\n".join(f"  {k}={v}" for k, v in sorted(env_vars.items()))
        (logs_dir / "cmd.txt").write_text(
            "\n".join(
                [
                    f"workdir: {self.settings.depthsplat_root}",
                    f"script: {script_path}",
                    f"env:",
                    env_lines,
                    f"command: bash {script_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        export_lines = "".join(
            f"export {key}={shlex.quote(value)}\n" for key, value in sorted(env_vars.items())
        )
        (logs_dir / "cmd.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + f"cd {shlex.quote(str(self.settings.depthsplat_root))}\n"
            + export_lines
            + f"bash {shlex.quote(script_path)}\n",
            encoding="utf-8",
        )
        os.chmod(logs_dir / "cmd.sh", 0o755)

    def start_and_wait(
        self,
        task_id: str,
        preset: PresetConfig,
        command: list[str],
        task_dir: Path,
    ) -> RunnerResult:
        stdout_path = task_dir / "logs" / "stdout.log"
        stderr_path = task_dir / "logs" / "stderr.log"
        env = os.environ.copy()
        env.setdefault("SKVIDEO_FFMPEG_PATH", "/usr/bin")
        env.update(preset.env)
        started_at = datetime.now(timezone.utc)
        start_monotonic = time.monotonic()
        cancel_event = Event()
        with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
            popen = subprocess.Popen(
                command,
                cwd=str(self.settings.depthsplat_root),
                env=env,
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
                preexec_fn=os.setsid,
            )
            pgid = os.getpgid(popen.pid)
            running = RunningProcess(popen=popen, task_id=task_id, pgid=pgid, cancel_event=cancel_event)
            with self._lock:
                self._active[task_id] = running
            self.logger.info("subprocess started", event="runner_started", fields={"task_id": task_id, "pid": popen.pid, "pgid": pgid})
            return_code: int | None = None
            cancelled = False
            try:
                while True:
                    return_code = popen.poll()
                    if return_code is not None:
                        break
                    if cancel_event.is_set():
                        cancelled = True
                    time.sleep(0.25)
            finally:
                with self._lock:
                    self._active.pop(task_id, None)
        finished_at = datetime.now(timezone.utc)
        runtime_seconds = time.monotonic() - start_monotonic
        return RunnerResult(
            return_code=return_code,
            started_at=started_at,
            finished_at=finished_at,
            runtime_seconds=runtime_seconds,
            cancelled=cancelled,
        )

    def start_and_wait_script(
        self,
        task_id: str,
        script_path: Path,
        env_vars: dict[str, str],
        task_dir: Path,
    ) -> RunnerResult:
        """执行脚本，通过环境变量传参"""
        stdout_path = task_dir / "logs" / "stdout.log"
        stderr_path = task_dir / "logs" / "stderr.log"
        env = os.environ.copy()
        env.update(env_vars)
        command = ["bash", str(script_path)]
        started_at = datetime.now(timezone.utc)
        start_monotonic = time.monotonic()
        cancel_event = Event()
        with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
            popen = subprocess.Popen(
                command,
                cwd=str(self.settings.depthsplat_root),
                env=env,
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
                preexec_fn=os.setsid,
            )
            pgid = os.getpgid(popen.pid)
            running = RunningProcess(popen=popen, task_id=task_id, pgid=pgid, cancel_event=cancel_event)
            with self._lock:
                self._active[task_id] = running
            self.logger.info("script started", event="runner_script_started", fields={"task_id": task_id, "pid": popen.pid, "pgid": pgid, "script": str(script_path)})
            return_code: int | None = None
            cancelled = False
            try:
                while True:
                    return_code = popen.poll()
                    if return_code is not None:
                        break
                    if cancel_event.is_set():
                        cancelled = True
                    time.sleep(0.25)
            finally:
                with self._lock:
                    self._active.pop(task_id, None)
        finished_at = datetime.now(timezone.utc)
        runtime_seconds = time.monotonic() - start_monotonic
        return RunnerResult(
            return_code=return_code,
            started_at=started_at,
            finished_at=finished_at,
            runtime_seconds=runtime_seconds,
            cancelled=cancelled,
        )

    def cancel(self, task_id: str, grace_seconds: float) -> tuple[bool, bool, str | None]:
        with self._lock:
            running = self._active.get(task_id)
        if running is None:
            return False, False, "No active subprocess found"

        kill_sent = False
        force_kill_sent = False
        error: str | None = None
        running.cancel_event.set()
        try:
            if running.pgid is not None:
                os.killpg(running.pgid, signal.SIGTERM)
            else:
                running.popen.terminate()
            kill_sent = True
            deadline = time.monotonic() + grace_seconds
            while time.monotonic() < deadline:
                if running.popen.poll() is not None:
                    return kill_sent, force_kill_sent, None
                time.sleep(0.2)
            if running.pgid is not None:
                os.killpg(running.pgid, signal.SIGKILL)
            else:
                running.popen.kill()
            force_kill_sent = True
            return kill_sent, force_kill_sent, None
        except ProcessLookupError:
            return kill_sent, force_kill_sent, None
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            self.logger.error("cancel failed", event="runner_cancel_failed", fields={"task_id": task_id, "error": error})
            return kill_sent, force_kill_sent, error
