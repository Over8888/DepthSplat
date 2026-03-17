from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models.task import CancelMetadata, RequestMetadata, ResultMetadata, TaskRecord, TimingInfo


class FilesystemStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.outputs_root.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id: str) -> Path:
        return self.settings.outputs_root / task_id

    def create_task_layout(self, task_id: str) -> dict[str, Path]:
        base = self.task_dir(task_id)
        layout = {
            "task": base,
            "input": base / "input",
            "render": base / "render",
            "depth": base / "depth",
            "logs": base / "logs",
            "meta": base / "meta",
        }
        for path in layout.values():
            path.mkdir(parents=True, exist_ok=True)
        return layout

    def meta_path(self, task_id: str, name: str) -> Path:
        return self.task_dir(task_id) / "meta" / name

    def log_path(self, task_id: str, name: str) -> Path:
        return self.task_dir(task_id) / "logs" / name

    def save_task(self, task: TaskRecord) -> TaskRecord:
        task.updated_at = datetime.now(timezone.utc)
        self.create_task_layout(task.id)
        self.meta_path(task.id, "task.json").write_text(task.model_dump_json(indent=2), encoding="utf-8")
        self.write_request(task.id, task.request)
        self.write_timing(task.id, task.timing)
        if task.result is not None:
            self.write_result(task.id, task.result)
        if task.cancel is not None:
            self.write_cancel(task.id, task.cancel)
        return task

    def load_task(self, task_id: str) -> TaskRecord:
        path = self.meta_path(task_id, "task.json")
        if not path.exists():
            raise FileNotFoundError(f"Task not found: {task_id}")
        return TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_tasks(self) -> list[TaskRecord]:
        tasks: list[TaskRecord] = []
        for path in self.settings.outputs_root.glob("*/meta/task.json"):
            try:
                tasks.append(TaskRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        tasks.sort(key=lambda item: item.created_at)
        return tasks

    def write_request(self, task_id: str, request: RequestMetadata) -> None:
        self.meta_path(task_id, "request.json").write_text(
            json.dumps(request.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_result(self, task_id: str, result: ResultMetadata) -> None:
        self.meta_path(task_id, "result.json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_timing(self, task_id: str, timing: TimingInfo) -> None:
        self.meta_path(task_id, "timing.json").write_text(
            json.dumps(timing.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_cancel(self, task_id: str, cancel: CancelMetadata) -> None:
        self.meta_path(task_id, "cancel.json").write_text(
            json.dumps(cancel.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def append_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)
            if text and not text.endswith("\n"):
                fh.write("\n")

    def read_log(self, task_id: str, stream: str, tail_lines: int | None = None) -> str:
        path = self.log_path(task_id, f"{stream}.log")
        if not path.exists():
            return ""
        data = path.read_text(encoding="utf-8", errors="ignore")
        if tail_lines is None or tail_lines <= 0:
            return data
        lines = data.splitlines()
        return "\n".join(lines[-tail_lines:])

    def artifact_url(self, task_id: str, relative_path: str) -> str:
        return f"/artifacts/{task_id}/{relative_path}"
