from __future__ import annotations

import shutil
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from app.config import PresetConfig, Settings
from app.models.enums import ALLOWED_TRANSITIONS, TERMINAL_STATES, TaskState
from app.models.task import CancelMetadata, ProcessInfo, RequestMetadata, TaskRecord, TimingInfo
from app.schemas.tasks import CreateTaskRequest
from app.services.result_builder import ResultBuilder
from app.services.runner import Runner
from app.services.sample_service import SampleService
from app.services.storage import FilesystemStorage
from app.utils.logging import get_logger


class TaskManager:
    def __init__(
        self,
        settings: Settings,
        storage: FilesystemStorage,
        sample_service: SampleService,
        runner: Runner,
        result_builder: ResultBuilder,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.sample_service = sample_service
        self.runner = runner
        self.result_builder = result_builder
        self.logger = get_logger(__name__)
        self._queue: deque[str] = deque()
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._shutdown = False
        self._worker = threading.Thread(target=self._worker_loop, name="depthsplat-v3-worker", daemon=True)
        self._current_task_id: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._worker.is_alive():
                return
            self._recover_incomplete_tasks()
            self._worker.start()
            self.logger.info("task manager started", event="task_manager_started", fields={})

    def shutdown(self) -> None:
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()

    def create_task(self, request: CreateTaskRequest, upload_temp_paths: list[str] | None = None) -> TaskRecord:
        preset_name = request.preset_id or self.settings.default_preset
        preset = self._preset(preset_name)
        sample = None
        if request.sample_id:
            sample = self.sample_service.get_sample(request.sample_id)
            if sample.preset != preset_name:
                raise ValueError("sample_id does not match preset")
        elif not upload_temp_paths:
            raise ValueError("images is required when sampleId is omitted")

        task_id = self._new_task_id()
        self.storage.create_task_layout(task_id)
        stored_images = self._persist_uploaded_images(task_id, upload_temp_paths or [], request.images)
        now = datetime.now(timezone.utc)
        task = TaskRecord(
            id=task_id,
            status=TaskState.queued,
            stage=TaskState.queued,
            created_at=now,
            updated_at=now,
            request=RequestMetadata(
                preset=preset.name,
                sample_id=request.sample_id,
                scene_key=sample.scene_key if sample is not None else f"upload_{task_id}",
                checkpoint=preset.checkpoint_path.name,
                image_shape=list(preset.image_shape),
                num_context_views=preset.num_context_views,
                images=stored_images,
                options=request.options,
                defaults=sample.defaults if sample is not None else {
                    "checkpoint": preset.checkpoint_path.name,
                    "num_context_views": preset.num_context_views,
                    "image_shape": list(preset.image_shape),
                },
            ),
            timing=TimingInfo(queued_at=now),
        )
        self.storage.save_task(task)
        with self._cond:
            self._queue.append(task_id)
            self._cond.notify_all()
        self.logger.info(
            "task created",
            event="task_created",
            fields={
                "task_id": task_id,
                "preset": preset.name,
                "sample_id": request.sample_id,
                "image_count": len(stored_images),
                "options": request.options,
            },
        )
        return task

    def get_task(self, task_id: str) -> TaskRecord:
        return self.storage.load_task(task_id)

    def cancel_task(self, task_id: str, reason: str, requested_by: str = "frontend") -> TaskRecord:
        with self._lock:
            task = self.storage.load_task(task_id)
            if task.status in TERMINAL_STATES:
                return task
            cancel = task.cancel or CancelMetadata(requested_at=datetime.now(timezone.utc), reason=reason, requested_by=requested_by)
            task.cancel = cancel
            if task.status == TaskState.queued:
                try:
                    self._queue.remove(task_id)
                except ValueError:
                    pass
                cancel.finalised_at = datetime.now(timezone.utc)
                cancel.outcome = "cancelled_while_queued"
                task = self._transition(task, TaskState.cancelled)
                task.result = self.result_builder.build_cancelled(task)
                self.storage.save_task(task)
                return task

            self.storage.save_task(task)
            self.logger.info("cancel requested", event="task_cancel_requested", fields={"task_id": task_id, "state": task.status.value})
            kill_sent, force_kill_sent, error = self.runner.cancel(task_id, self.settings.cancellation_grace_seconds)
            task = self.storage.load_task(task_id)
            task.cancel = task.cancel or cancel
            task.cancel.kill_sent = kill_sent
            task.cancel.force_kill_sent = force_kill_sent
            if error == "No active subprocess found":
                task.cancel.error = None
                if task_id != self._current_task_id:
                    task.cancel.finalised_at = datetime.now(timezone.utc)
                    task.cancel.outcome = "cancelled_without_active_process"
                    task = self._transition(task, TaskState.cancelled)
                    task.timing.finished_at = task.cancel.finalised_at
                    task.timing.total_seconds = self._duration(task.timing.queued_at, task.timing.finished_at)
                    task.result = self.result_builder.build_cancelled(task)
                    self.storage.save_task(task)
                    return task
                self.storage.save_task(task)
                return task
            task.cancel.error = error
            if error:
                task.error_summary = f"Cancellation error: {error}"
                self.storage.save_task(task)
                raise RuntimeError(error)
            return task

    def list_samples(self, preset_name: str | None = None):
        preset_name = preset_name or self.settings.default_preset
        return self.sample_service.list_samples(preset_name)

    def list_presets(self):
        return self.sample_service.list_presets()

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._shutdown:
                    self._cond.wait(timeout=1.0)
                if self._shutdown:
                    return
                task_id = self._queue.popleft()
                self._current_task_id = task_id
            try:
                self._run_task(task_id)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("task worker exception", event="task_worker_exception", fields={"task_id": task_id, "error": str(exc)})
                try:
                    task = self.storage.load_task(task_id)
                    if task.status not in TERMINAL_STATES:
                        task.error_summary = str(exc)
                        task = self._transition(task, TaskState.failed)
                        task.timing.finished_at = datetime.now(timezone.utc)
                        task.timing.total_seconds = self._duration(task.timing.queued_at, task.timing.finished_at)
                        self.storage.save_task(task)
                except Exception:
                    pass
            finally:
                with self._lock:
                    self._current_task_id = None

    def _run_task(self, task_id: str) -> None:
        task = self.storage.load_task(task_id)
        if task.status in TERMINAL_STATES:
            return
        preset = self._preset(task.request.preset)

        task = self._transition(task, TaskState.preparing)
        task.timing.preparing_started_at = datetime.now(timezone.utc)
        self.storage.save_task(task)

        task_dir = self.storage.task_dir(task_id)
        prep_start = time.monotonic()
        if task.request.sample_id:
            materialized = self.sample_service.materialize_sample(task_dir, task.request.sample_id, preset.name)
        else:
            materialized = self.sample_service.materialize_uploaded_images(task_dir, task.request.images, task.request.scene_key)
        data_prep_seconds = time.monotonic() - prep_start
        task.timing.data_prep_seconds = round(data_prep_seconds, 4)
        task.timing.startup_seconds = self._duration(task.timing.queued_at, task.timing.preparing_started_at)
        self.storage.save_task(task)

        if not task.request.sample_id:
            task.error_summary = "Manual upload task creation is supported, but raw uploaded images cannot be sent to DepthSplat inference without dataset/camera metadata."
            task.timing.finished_at = datetime.now(timezone.utc)
            task.timing.total_seconds = self._duration(task.timing.queued_at, task.timing.finished_at)
            task = self._transition(task, TaskState.failed)
            self.storage.save_task(task)
            return

        output_dir = task_dir / "meta" / "depthsplat_output"
        command = self.runner.build_command(
            preset,
            materialized["dataset_root"],
            materialized["evaluation_index_path"],
            output_dir,
            task.request.options,
        )
        self.runner.write_command_files(task_dir, preset, command)
        task.process = ProcessInfo(command=command)
        self.storage.save_task(task)

        if task.cancel is not None:
            task.cancel.finalised_at = datetime.now(timezone.utc)
            task.cancel.outcome = "cancelled_before_run"
            task = self._transition(task, TaskState.cancelled)
            task.timing.finished_at = task.cancel.finalised_at
            task.timing.total_seconds = self._duration(task.timing.queued_at, task.timing.finished_at)
            task.result = self.result_builder.build_cancelled(task)
            self.storage.save_task(task)
            return

        task = self._transition(task, TaskState.running)
        task.timing.running_started_at = datetime.now(timezone.utc)
        self.storage.save_task(task)
        runner_result = self.runner.start_and_wait(task_id, preset, command, task_dir)

        task = self.storage.load_task(task_id)
        task.process.pid = None
        task.process.pgid = None

        if task.cancel is not None or runner_result.cancelled:
            if task.cancel is None:
                task.cancel = CancelMetadata(requested_at=runner_result.finished_at, reason="process_cancelled")
            task.cancel.finalised_at = runner_result.finished_at
            task.cancel.outcome = "cancelled_during_run"
            task.timing.finished_at = runner_result.finished_at
            task.timing.total_seconds = self._duration(task.timing.queued_at, task.timing.finished_at)
            task = self._transition(task, TaskState.cancelled)
            task.result = self.result_builder.build_cancelled(task)
            self.storage.save_task(task)
            return

        if runner_result.return_code != 0:
            task.error_summary = f"DepthSplat exited with code {runner_result.return_code}"
            task.timing.finished_at = runner_result.finished_at
            task.timing.total_seconds = self._duration(task.timing.queued_at, task.timing.finished_at)
            task = self._transition(task, TaskState.failed)
            self.storage.save_task(task)
            return

        task = self._transition(task, TaskState.postprocessing)
        task.timing.postprocessing_started_at = datetime.now(timezone.utc)
        post_start = time.monotonic()
        result, timing_patch = self.result_builder.build(task)
        task.timing.finished_at = datetime.now(timezone.utc)
        task.timing.save_outputs_seconds = round(time.monotonic() - post_start, 4)
        for key, value in timing_patch.items():
            setattr(task.timing, key, value)
        task.timing.total_seconds = self._duration(task.timing.queued_at, task.timing.finished_at)
        task.result = result
        task = self._transition(task, TaskState.success)
        if task.result is not None:
            task.result.status = task.status
            task.result.result_complete = True
        self.storage.save_task(task)

    def _recover_incomplete_tasks(self) -> None:
        for task in self.storage.list_tasks():
            if task.status in TERMINAL_STATES:
                continue
            task.error_summary = "Recovered on startup before rescheduling"
            task = self._transition(task, TaskState.failed, allow_same=True)
            task.timing.finished_at = datetime.now(timezone.utc)
            task.timing.total_seconds = self._duration(task.timing.queued_at, task.timing.finished_at)
            self.storage.save_task(task)

    def _transition(self, task: TaskRecord, new_state: TaskState, allow_same: bool = False) -> TaskRecord:
        if task.status == new_state and allow_same:
            return task
        if task.status in TERMINAL_STATES:
            raise ValueError(f"Cannot transition terminal task {task.id} from {task.status} to {new_state}")
        if new_state not in ALLOWED_TRANSITIONS[task.status]:
            raise ValueError(f"Invalid transition for task {task.id}: {task.status} -> {new_state}")
        task.status = new_state
        task.stage = new_state
        task.updated_at = datetime.now(timezone.utc)
        return task

    def _preset(self, preset_name: str) -> PresetConfig:
        if preset_name not in self.settings.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        return self.settings.presets[preset_name]

    def _new_task_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]

    def _persist_uploaded_images(self, task_id: str, upload_temp_paths: list[str], image_names: list[str]) -> list[str]:
        if not upload_temp_paths:
            return []
        input_dir = self.storage.task_dir(task_id) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        stored: list[str] = []
        for index, temp_path in enumerate(upload_temp_paths):
            src = Path(temp_path)
            suffix = src.suffix.lower()
            if not suffix:
                original_name = image_names[index] if index < len(image_names) else ""
                if "." in original_name:
                    suffix = "." + original_name.rsplit(".", 1)[-1].lower()
            filename = f"upload_{index:02d}{suffix or '.png'}"
            dst = input_dir / filename
            shutil.copy2(src, dst)
            stored.append(str(dst))
            try:
                src.unlink(missing_ok=True)
            except Exception:
                pass
        return stored

    @staticmethod
    def _duration(start: datetime | None, end: datetime | None) -> float | None:
        if start is None or end is None:
            return None
        return round((end - start).total_seconds(), 4)
