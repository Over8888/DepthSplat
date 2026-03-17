from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import ValidationError

from app.models.enums import TaskState
from app.schemas.tasks import (
    CancelTaskRequest,
    CancelTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    TaskLogsResponse,
    TaskResponse,
    TaskResultResponse,
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
    from pathlib import Path

    task_dir = Path(request.app.state.storage.task_dir(task_id))
    preview_dir = task_dir / "input" / "preview"
    if preview_dir.exists():
        return _absolute_urls(
            request,
            [f"/artifacts/{task_id}/input/preview/{path.name}" for path in sorted(preview_dir.glob("*.png"))],
        )
    return _absolute_urls(
        request,
        [f"/artifacts/{task_id}/input/{path.name}" for path in sorted((task_dir / "input").glob("upload_*"))],
    )


def _task_stage(task, input_images: list[str]) -> str:
    if task.status == TaskState.preparing and input_images:
        return "data_prep"
    return task.stage.value


def _sample_name(manager, task) -> str | None:
    try:
        return manager.sample_service.get_sample(task.request.sample_id).label
    except Exception:
        return None


def _preset_name(manager, task) -> str | None:
    try:
        return manager.settings.presets[task.request.preset].display_name
    except Exception:
        return None


def _build_task_response_payload(request: Request, task) -> dict:
    manager = request.app.state.task_manager
    input_images = _task_input_images(request, task.id)
    return {
        "task_id": task.id,
        "status": task.status,
        "stage": _task_stage(task, input_images),
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "input_images": input_images,
        "sample_id": task.request.sample_id,
        "sample_name": _sample_name(manager, task),
        "preset_id": task.request.preset,
        "preset_name": _preset_name(manager, task),
    }


async def _parse_create_task_payload(request: Request) -> CreateTaskRequest:
    content_type = (request.headers.get("content-type") or "").lower()
    payload: dict = {}
    if "application/json" in content_type:
        payload = await request.json()
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        images: list[str] = []
        for key, value in form.multi_items():
            if key in {"images", "images[]"}:
                filename = getattr(value, "filename", None)
                images.append(filename or str(value))
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
                    images.append(filename or str(value))
                else:
                    payload[key] = value
            if images:
                payload["images"] = images
    try:
        return CreateTaskRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@router.post("/tasks", response_model=CreateTaskResponse)
async def create_task(request: Request) -> CreateTaskResponse:
    manager = request.app.state.task_manager
    payload = await _parse_create_task_payload(request)
    try:
        task = manager.create_task(payload)
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
def cancel_task(task_id: str, payload: CancelTaskRequest, request: Request) -> CancelTaskResponse:
    manager = request.app.state.task_manager
    try:
        task = manager.cancel_task(task_id, payload.reason)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CancelTaskResponse(task_id=task.id, status=task.status, cancel=task.cancel, error_summary=task.error_summary)


@router.get("/tasks/{task_id}/logs", response_model=TaskLogsResponse)
def get_logs(task_id: str, request: Request, tail_lines: int | None = Query(default=None, ge=1, le=5000)) -> TaskLogsResponse:
    manager = request.app.state.task_manager
    storage = request.app.state.storage
    try:
        task = manager.get_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    tail = tail_lines or manager.settings.log_tail_lines
    return TaskLogsResponse(
        task_id=task.id,
        status=task.status,
        stage=_task_stage(task, _task_input_images(request, task.id)),
        stdout=storage.read_log(task_id, "stdout", tail),
        stderr=storage.read_log(task_id, "stderr", tail),
    )


@router.get("/tasks/{task_id}/result", response_model=TaskResultResponse)
def get_result(task_id: str, request: Request) -> TaskResultResponse:
    manager = request.app.state.task_manager
    try:
        task = manager.get_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = _build_task_response_payload(request, task)
    payload["video_url"] = _absolute_url(request, task.result.video_url) if task.result is not None else None
    return TaskResultResponse(**payload)
