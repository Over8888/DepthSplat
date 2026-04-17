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
