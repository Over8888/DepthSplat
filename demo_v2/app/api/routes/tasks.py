from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import ValidationError

from app.models.enums import TaskState
from app.schemas.tasks import (
    CancelTaskRequest,
    CancelTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    TaskLogEntry,
    TaskLogsResponse,
    TaskResponse,
    TaskResultResponse,
    TaskTimings,
)

router = APIRouter(tags=["tasks"])


def _absolute_url(request: Request, path: str | None) -> str | None:
    if path is None:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        return path
    return str(request.base_url).rstrip("/") + path


def _absolute_urls(request: Request, paths: list[str] | None) -> list[str]:
    if not paths:
        return []
    return [item for item in (_absolute_url(request, path) for path in paths) if item is not None]


def _task_input_images(request: Request, task_id: str) -> list[str]:
    task_dir = Path(request.app.state.storage.task_dir(task_id))
    preview_dir = task_dir / "input" / "preview"
    if preview_dir.exists():
        preview_paths = sorted(preview_dir.glob("*.png")) + sorted(preview_dir.glob("*.jpg")) + sorted(preview_dir.glob("*.jpeg"))
        return _absolute_urls(request, [f"/artifacts/{task_id}/input/preview/{path.name}" for path in preview_paths])
    upload_paths = sorted((task_dir / "input").glob("upload_*"))
    return _absolute_urls(request, [f"/artifacts/{task_id}/input/{path.name}" for path in upload_paths])


def _task_stage(task, input_images: list[str]) -> str:
    if task.status == TaskState.preparing and input_images:
        return "data_prep"
    return task.stage.value


def _sample_name(manager, task) -> str | None:
    if not task.request.sample_id:
        return None
    try:
        return manager.sample_service.get_sample(task.request.sample_id).label
    except Exception:
        return None


def _preset_name(manager, task) -> str | None:
    try:
        return manager.settings.presets[task.request.preset].display_name
    except Exception:
        return None


def _seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 4)


def _task_timings(task) -> TaskTimings:
    timing = task.timing
    started_at = timing.running_started_at or timing.preparing_started_at
    updated_at = task.updated_at
    duration_anchor = started_at or timing.queued_at
    duration_seconds = _seconds_between(duration_anchor, timing.finished_at or updated_at)

    inference_seconds = timing.forward_seconds
    if (
        inference_seconds is None
        and timing.running_started_at is not None
        and timing.postprocessing_started_at is not None
    ):
        inference_seconds = _seconds_between(timing.running_started_at, timing.postprocessing_started_at)

    return TaskTimings(
        queued_at=timing.queued_at,
        started_at=started_at,
        updated_at=updated_at,
        duration_seconds=duration_seconds,
        data_load_seconds=timing.model_load_seconds,
        data_prep_seconds=timing.data_prep_seconds,
        inference_seconds=inference_seconds,
        splat_conversion_seconds=timing.splat_conversion_seconds,
    )


def _build_task_response_payload(request: Request, task) -> dict:
    manager = request.app.state.task_manager
    input_images = _task_input_images(request, task.id)
    return {
        "task_id": task.id,
        "status": task.status,
        "stage": _task_stage(task, input_images),
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "timings": _task_timings(task),
        "input_images": input_images,
        "sample_id": task.request.sample_id,
        "sample_name": _sample_name(manager, task),
        "preset_id": task.request.preset,
        "preset_name": _preset_name(manager, task),
    }


_ISO_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))")


def _parse_log_timestamp(line: str, fallback: datetime) -> datetime:
    text = line.strip()
    if not text:
        return fallback
    if text.startswith("{"):
        try:
            payload = json.loads(text)
            ts = payload.get("ts") or payload.get("timestamp")
            if isinstance(ts, str):
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            pass
    match = _ISO_TS_RE.search(text)
    if match:
        try:
            return datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
        except Exception:
            pass
    return fallback


def _infer_log_level(line: str, stream: str) -> str:
    lowered = line.lower()
    if stream == "stderr":
        if "warning" in lowered:
            return "warning"
        return "error"
    if '"level": "error"' in lowered or "traceback" in lowered:
        return "error"
    if '"level": "warning"' in lowered or "warning" in lowered:
        return "warning"
    return "info"


def _lifecycle_entries(task) -> list[tuple[datetime, str, str]]:
    entries: list[tuple[datetime, str, str]] = []
    if task.timing.queued_at is not None:
        entries.append((task.timing.queued_at, "info", "task queued"))
    if task.timing.preparing_started_at is not None:
        entries.append((task.timing.preparing_started_at, "info", "data prep started"))
    if task.timing.running_started_at is not None:
        entries.append((task.timing.running_started_at, "info", "data prep finished"))
        entries.append((task.timing.running_started_at, "info", "inference running"))
    if task.timing.postprocessing_started_at is not None:
        entries.append((task.timing.postprocessing_started_at, "info", "postprocessing started"))
    if task.timing.finished_at is not None:
        if task.status == TaskState.success:
            entries.append((task.timing.finished_at, "info", "task finished successfully"))
        elif task.status == TaskState.cancelled:
            entries.append((task.timing.finished_at, "warning", "task cancelled"))
        elif task.status == TaskState.failed:
            entries.append((task.timing.finished_at, "error", task.error_summary or "task failed"))
    return entries


def _task_log_entries(request: Request, task_id: str, tail_lines: int) -> list[TaskLogEntry]:
    manager = request.app.state.task_manager
    storage = request.app.state.storage
    task = manager.get_task(task_id)
    entries: list[TaskLogEntry] = []
    counter = 1

    for timestamp, level, message in _lifecycle_entries(task):
        entries.append(TaskLogEntry(id=str(counter), timestamp=timestamp, level=level, message=message))
        counter += 1

    fallback_ts = task.updated_at or task.created_at or datetime.now(timezone.utc)
    for stream in ("stdout", "stderr"):
        raw = storage.read_log(task_id, stream, tail_lines)
        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            entries.append(
                TaskLogEntry(
                    id=str(counter),
                    timestamp=_parse_log_timestamp(text, fallback_ts),
                    level=_infer_log_level(text, stream),
                    message=text,
                )
            )
            counter += 1

    entries.sort(key=lambda item: (item.timestamp, int(item.id)))
    for index, entry in enumerate(entries, start=1):
        entry.id = str(index)
    return entries


async def _parse_create_task_payload(request: Request) -> tuple[CreateTaskRequest, list[str]]:
    content_type = (request.headers.get("content-type") or "").lower()
    payload: dict = {}
    upload_temp_paths: list[str] = []
    if "application/json" in content_type:
        payload = await request.json()
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        images: list[str] = []
        for key, value in form.multi_items():
            if key in {"images", "images[]"}:
                filename = getattr(value, "filename", None)
                if filename is not None and hasattr(value, "read"):
                    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
                    with NamedTemporaryFile(prefix="depthsplat_v3_", suffix=suffix, delete=False) as tmp:
                        tmp.write(await value.read())
                        upload_temp_paths.append(tmp.name)
                    images.append(filename)
                else:
                    images.append(str(value))
            else:
                payload[key] = value
        if images:
            payload["images"] = images
    else:
        try:
            payload = await request.json()
        except Exception:
            form = await request.form()
            images: list[str] = []
            for key, value in form.multi_items():
                if key in {"images", "images[]"}:
                    filename = getattr(value, "filename", None)
                    if filename is not None and hasattr(value, "read"):
                        suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
                        with NamedTemporaryFile(prefix="depthsplat_v3_", suffix=suffix, delete=False) as tmp:
                            tmp.write(await value.read())
                            upload_temp_paths.append(tmp.name)
                        images.append(filename)
                    else:
                        images.append(str(value))
                else:
                    payload[key] = value
            if images:
                payload["images"] = images
    try:
        parsed = CreateTaskRequest.model_validate(payload)
        return parsed, upload_temp_paths
    except ValidationError as exc:
        for temp_path in upload_temp_paths:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@router.post("/tasks", response_model=CreateTaskResponse)
async def create_task(request: Request) -> CreateTaskResponse:
    manager = request.app.state.task_manager
    payload, upload_temp_paths = await _parse_create_task_payload(request)
    try:
        task = manager.create_task(payload, upload_temp_paths)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateTaskResponse(id=task.id, state=task.status, created_at=task.created_at)


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, request: Request) -> TaskResponse:
    manager = request.app.state.task_manager
    try:
        task = manager.get_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TaskResponse(**_build_task_response_payload(request, task))


@router.post("/tasks/{task_id}/cancel", response_model=CancelTaskResponse)
def cancel_task(task_id: str, request: Request, payload: CancelTaskRequest | None = Body(default=None)) -> CancelTaskResponse:
    manager = request.app.state.task_manager
    try:
        reason = payload.reason if payload is not None else "frontend_cancel_request"
        task = manager.cancel_task(task_id, reason)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CancelTaskResponse(task_id=task.id, status=task.status, cancel=task.cancel, error_summary=task.error_summary)


@router.get("/tasks/{task_id}/logs", response_model=TaskLogsResponse)
def get_logs(task_id: str, request: Request, tail_lines: int | None = Query(default=None, ge=1, le=5000)) -> TaskLogsResponse:
    manager = request.app.state.task_manager
    try:
        manager.get_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    tail = tail_lines or manager.settings.log_tail_lines
    return TaskLogsResponse(task_id=task_id, entries=_task_log_entries(request, task_id, tail))


@router.get("/tasks/{task_id}/result", response_model=TaskResultResponse)
def get_result(task_id: str, request: Request) -> TaskResultResponse:
    manager = request.app.state.task_manager
    try:
        task = manager.get_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = _build_task_response_payload(request, task)
    result = task.result
    payload["video_url"] = _absolute_url(request, result.video_url) if result is not None else None
    payload["depth_images"] = _absolute_urls(request, result.depth_images) if result is not None else []
    payload["splat_url"] = _absolute_url(request, result.splat_url) if result is not None else None
    return TaskResultResponse(**payload)
