from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.models.enums import TaskState
from app.utils.timezone import app_timezone, to_local
from app.schemas.tasks import (
    CancelTaskRequest,
    CancelTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    InputImageItem,
    InputImagesResponse,
    TaskLogEntry,
    TaskLogsResponse,
    TaskResponse,
    TaskResultResponse,
    TaskTimings,
)

router = APIRouter(tags=["tasks"])


VIDEO_UPLOAD_KEYS = {"video", "video[]", "videoFile", "video_file"}
POSE_UPLOAD_KEYS = {"pose_file", "poseFile", "camera_file", "cameraFile"}
IMAGE_UPLOAD_KEYS = {"images", "images[]"}


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
        "message": None,
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
_TASK_ID_DATE_RE = re.compile(r"^(\d{8})_(\d{6})_[0-9a-f]{8}$")
_TASK_ID_LOCAL_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_.+_[0-9a-f]{8}$")


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
        entries.append(TaskLogEntry(id=str(counter), timestamp=to_local(timestamp), level=level, message=message))
        counter += 1

    task_tz = app_timezone()
    fallback_ts = to_local(task.updated_at or task.created_at or datetime.now(task_tz))
    for stream in ("stdout", "stderr"):
        raw = storage.read_log(task_id, stream, tail_lines)
        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            entries.append(
                TaskLogEntry(
                    id=str(counter),
                    timestamp=to_local(_parse_log_timestamp(text, fallback_ts)),
                    level=_infer_log_level(text, stream),
                    message=text,
                )
            )
            counter += 1

    entries.sort(key=lambda item: (item.timestamp, int(item.id)))
    for index, entry in enumerate(entries, start=1):
        entry.id = str(index)
    return entries


def _task_id_timestamp(task_id: str, task_timezone: ZoneInfo) -> datetime:
    match = _TASK_ID_DATE_RE.match(task_id)
    if match:
        date_part, time_part = match.groups()
        try:
            return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    match = _TASK_ID_LOCAL_RE.match(task_id)
    if match:
        date_part, time_part = match.groups()
        try:
            naive = datetime.strptime(f"{date_part}_{time_part}", "%Y-%m-%d_%H-%M-%S")
            return naive.replace(tzinfo=task_timezone)
        except ValueError:
            pass

    return datetime.now(task_timezone)


def _missing_task_body(request: Request, task_id: str) -> dict:
    task_tz = ZoneInfo(request.app.state.settings.task_id_timezone)
    timestamp = _task_id_timestamp(task_id, task_tz).astimezone(task_tz)
    return {
        "task_id": task_id,
        "status": "missing",
        "stage": "not_found",
        "updated_at": timestamp.isoformat(),
    }


def _missing_task_response(request: Request, task_id: str) -> JSONResponse:
    return JSONResponse(status_code=404, content=_missing_task_body(request, task_id))


async def _save_upload_to_temp(upload) -> tuple[str, str]:
    filename = getattr(upload, "filename", None)
    if filename is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=422, detail="Uploaded file is invalid")
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    with NamedTemporaryFile(prefix="depthsplat_v3_", suffix=suffix, delete=False) as tmp:
        tmp.write(await upload.read())
        return tmp.name, filename


async def _parse_form_payload(request: Request) -> tuple[dict, list[str], str | None, str | None, str | None]:
    form = await request.form()
    payload: dict = {}
    upload_temp_paths: list[str] = []
    upload_temp_video_path: str | None = None
    upload_temp_pose_path: str | None = None
    uploaded_pose_name: str | None = None
    images: list[str] = []

    for key, value in form.multi_items():
        if key in IMAGE_UPLOAD_KEYS:
            filename = getattr(value, "filename", None)
            if filename is not None and hasattr(value, "read"):
                temp_path, stored_name = await _save_upload_to_temp(value)
                upload_temp_paths.append(temp_path)
                images.append(stored_name)
            else:
                images.append(str(value))
            continue

        if key in VIDEO_UPLOAD_KEYS:
            if upload_temp_video_path is not None:
                raise HTTPException(status_code=422, detail="Only one video file may be uploaded per task")
            if getattr(value, "filename", None) is not None and hasattr(value, "read"):
                temp_path, stored_name = await _save_upload_to_temp(value)
                upload_temp_video_path = temp_path
                payload["video"] = stored_name
            else:
                payload["video"] = str(value)
            continue

        if key in POSE_UPLOAD_KEYS:
            if upload_temp_pose_path is not None:
                raise HTTPException(status_code=422, detail="Only one camera/pose file may be uploaded per task")
            if getattr(value, "filename", None) is not None and hasattr(value, "read"):
                temp_path, stored_name = await _save_upload_to_temp(value)
                upload_temp_pose_path = temp_path
                uploaded_pose_name = stored_name
            else:
                payload["pose_file"] = str(value)
            continue

        payload[key] = value

    if images:
        payload["images"] = images
    if uploaded_pose_name and upload_temp_pose_path:
        payload["pose_file"] = uploaded_pose_name
    return payload, upload_temp_paths, upload_temp_video_path, upload_temp_pose_path, uploaded_pose_name


async def _parse_create_task_payload(request: Request) -> tuple[CreateTaskRequest, list[str], str | None, str | None, str | None]:
    content_type = (request.headers.get("content-type") or "").lower()
    payload: dict = {}
    upload_temp_paths: list[str] = []
    upload_temp_video_path: str | None = None
    upload_temp_pose_path: str | None = None
    uploaded_pose_name: str | None = None

    try:
        if "application/json" in content_type:
            payload = await request.json()
        elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            payload, upload_temp_paths, upload_temp_video_path, upload_temp_pose_path, uploaded_pose_name = await _parse_form_payload(request)
        else:
            try:
                payload = await request.json()
            except Exception:
                payload, upload_temp_paths, upload_temp_video_path, upload_temp_pose_path, uploaded_pose_name = await _parse_form_payload(request)

        parsed = CreateTaskRequest.model_validate(payload)
        return parsed, upload_temp_paths, upload_temp_video_path, upload_temp_pose_path, uploaded_pose_name
    except ValidationError as exc:
        for temp_path in upload_temp_paths:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
        if upload_temp_video_path is not None:
            try:
                Path(upload_temp_video_path).unlink(missing_ok=True)
            except Exception:
                pass
        if upload_temp_pose_path is not None:
            try:
                Path(upload_temp_pose_path).unlink(missing_ok=True)
            except Exception:
                pass
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@router.post("/tasks", response_model=CreateTaskResponse)
async def create_task(request: Request) -> CreateTaskResponse:
    manager = request.app.state.task_manager
    payload, upload_temp_paths, upload_temp_video_path, upload_temp_pose_path, uploaded_pose_name = await _parse_create_task_payload(request)
    try:
        task = manager.create_task(payload, upload_temp_paths, upload_temp_video_path, upload_temp_pose_path, uploaded_pose_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateTaskResponse(id=task.id, state=task.status, created_at=task.created_at)


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, request: Request) -> TaskResponse:
    manager = request.app.state.task_manager
    try:
        task = manager.get_task(task_id)
    except FileNotFoundError:
        return _missing_task_response(request, task_id)
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
    except FileNotFoundError:
        return _missing_task_response(request, task_id)
    tail = tail_lines or manager.settings.log_tail_lines
    return TaskLogsResponse(task_id=task_id, entries=_task_log_entries(request, task_id, tail))


@router.get("/tasks/{task_id}/result", response_model=TaskResultResponse)
def get_result(task_id: str, request: Request) -> TaskResultResponse:
    manager = request.app.state.task_manager
    try:
        task = manager.get_task(task_id)
    except FileNotFoundError:
        return _missing_task_response(request, task_id)
    payload = _build_task_response_payload(request, task)
    storage = request.app.state.storage
    result = task.result
    payload["video_url"] = _absolute_url(request, result.video_url) if result is not None else None
    payload["depth_images"] = _absolute_urls(request, result.depth_images) if result is not None else []
    payload["splat_url"] = _absolute_url(request, result.splat_url) if result is not None else None
    payload["rendered_images"] = _absolute_urls(request, result.rendered_images) if result is not None else []
    payload["gt_images"] = _absolute_urls(request, result.gt_images) if result is not None else []
    payload["error_images"] = _absolute_urls(request, result.error_images) if result is not None else []
    payload["comparison_concat_image"] = _absolute_url(request, result.comparison_concat_image) if result is not None else None
    payload["images_zip_url"] = _absolute_url(request, result.images_zip_url) if result is not None else None
    payload["camera_params_url"] = _absolute_url(request, result.camera_params_url) if result is not None else None
    payload["parameters"] = result.parameters if result is not None else {}

    # scores: merge from result + total_seconds from timing
    scores = dict(result.scores) if result is not None else {}
    timing_path = storage.task_dir(task_id) / "meta" / "timing.json"
    if timing_path.exists():
        try:
            timing_data = json.loads(timing_path.read_text(encoding="utf-8"))
            if "total_seconds" in timing_data and timing_data["total_seconds"] is not None:
                scores["total_seconds"] = timing_data["total_seconds"]
        except json.JSONDecodeError:
            pass
    payload["scores"] = scores

    # camera_intrinsics / camera_extrinsics from camera_params.json
    camera_intrinsics: list[dict] = []
    camera_extrinsics: list[dict] = []
    camera_params_path = storage.task_dir(task_id) / "meta" / "camera_params.json"
    if camera_params_path.exists():
        try:
            cam_data = json.loads(camera_params_path.read_text(encoding="utf-8"))
            for item in cam_data.get("images", []):
                camera_intrinsics.append({
                    "fx": item.get("fx", 0),
                    "fy": item.get("fy", 0),
                    "cx": item.get("cx", 0),
                    "cy": item.get("cy", 0),
                })
                camera_extrinsics.append({
                    "rotation": item.get("rotation", []),
                    "translation": item.get("translation", []),
                })
        except json.JSONDecodeError:
            pass
    payload["camera_intrinsics"] = camera_intrinsics
    payload["camera_extrinsics"] = camera_extrinsics

    return TaskResultResponse(**payload)


@router.get("/tasks/{task_id}/input-images", response_model=InputImagesResponse)
def get_input_images(task_id: str, request: Request) -> InputImagesResponse:
    manager = request.app.state.task_manager
    storage = request.app.state.storage

    try:
        task = manager.get_task(task_id)
    except FileNotFoundError:
        return _missing_task_response(request, task_id)

    eval_index_path = storage.task_dir(task_id) / "meta" / "evaluation_index.json"
    camera_params_path = storage.task_dir(task_id) / "meta" / "camera_params.json"

    context_indices: list[int] = []
    target_indices: list[int] = []
    scene_key = ""

    if eval_index_path.exists():
        try:
            eval_data = json.loads(eval_index_path.read_text(encoding="utf-8"))
            scene_key = next(iter(eval_data))
            eval_info = eval_data[scene_key]
            context_indices = eval_info.get("context", [])
            target_indices = eval_info.get("target", [])
        except (StopIteration, json.JSONDecodeError):
            pass

    all_indices = sorted(set(context_indices + target_indices))
    context_set = set(context_indices)

    camera_params_by_index: dict[int, dict] = {}
    if camera_params_path.exists():
        try:
            camera_data = json.loads(camera_params_path.read_text(encoding="utf-8"))
            for item in camera_data.get("images", []):
                camera_params_by_index[item["index"]] = item
        except json.JSONDecodeError:
            pass

    preview_dir = storage.task_dir(task_id) / "input" / "preview"
    preview_paths_by_index: dict[int, Path] = {}
    if preview_dir.exists():
        for path in sorted(preview_dir.glob("*.png")) + sorted(preview_dir.glob("*.jpg")) + sorted(preview_dir.glob("*.jpeg")):
            if not path.is_file():
                continue
            frame_index = _parse_frame_index_from_filename(path.stem)
            if frame_index is not None:
                preview_paths_by_index[frame_index] = path

    images: list[InputImageItem] = []
    for index in all_indices:
        is_context = index in context_set

        url = ""
        if is_context and index in preview_paths_by_index:
            url = _absolute_url(
                request,
                f"/artifacts/{task_id}/input/preview/{preview_paths_by_index[index].name}",
            )

        cam = camera_params_by_index.get(index)

        camera_params_status: Literal["available", "missing"] = "missing"
        intrinsics = None
        extrinsics = None
        width = None
        height = None

        if cam is not None:
            camera_params_status = "available"
            if all(k in cam for k in ("fx", "fy", "cx", "cy")):
                intrinsics = {
                    "fx": cam["fx"],
                    "fy": cam["fy"],
                    "cx": cam["cx"],
                    "cy": cam["cy"],
                }
            if "rotation" in cam and "translation" in cam:
                extrinsics = {
                    "rotation": cam["rotation"],
                    "translation": cam["translation"],
                }
            width = cam.get("width")
            height = cam.get("height")

        images.append(
            InputImageItem(
                url=url,
                index=index,
                width=width,
                height=height,
                isContextView=is_context,
                participatesInInference=True,
                cameraIntrinsics=intrinsics,
                cameraExtrinsics=extrinsics,
                cameraParamsStatus=camera_params_status,
            )
        )

    return InputImagesResponse(taskId=task_id, images=images)


def _parse_frame_index_from_filename(stem: str) -> int | None:
    match = re.search(r"(\d{6,})$", stem)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None
