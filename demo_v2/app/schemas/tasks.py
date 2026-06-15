from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import TaskState
from app.models.task import CancelMetadata


DEFAULT_TASK_OPTIONS: dict[str, Any] = {
    "testChunkInterval": True,
    "saveVideo": True,
    "computeScores": False,
    "exportDepthMap": True,
}


TASK_OPTION_ALIASES: dict[str, str] = {
    "test_chunk_interval": "testChunkInterval",
    "save_video": "saveVideo",
    "compute_scores": "computeScores",
    "export_depth_map": "exportDepthMap",
    "save_image": "saveImage",
    "save_gt_image": "saveGtImage",
    "save_input_images": "saveInputImages",
    "save_gaussian": "saveGaussian",
    "save_depth": "saveDepth",
    "save_depth_concat_img": "saveDepthConcatImg",
    "save_depth_npy": "saveDepthNpy",
    "save_ply": "savePly",
    "render_chunk_size": "renderChunkSize",
    "metric_chunk_size": "metricChunkSize",
}


def normalize_task_options(options: dict[str, Any]) -> dict[str, Any]:
    return {TASK_OPTION_ALIASES.get(key, key): value for key, value in options.items()}


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sample_id: str | None = Field(default=None, validation_alias=AliasChoices("sampleId", "sample_id"), serialization_alias="sampleId")
    preset_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("presetId", "preset"),
        serialization_alias="presetId",
    )
    images: list[str] = Field(default_factory=list)
    video: str | None = Field(default=None, validation_alias=AliasChoices("video", "videoFile", "video_file"))
    pose_file: str | None = Field(
        default=None,
        validation_alias=AliasChoices("poseFile", "pose_file", "cameraFile", "camera_file"),
        serialization_alias="poseFile",
    )
    pose_format: str | None = Field(
        default=None,
        validation_alias=AliasChoices("poseFormat", "pose_format"),
        serialization_alias="poseFormat",
    )
    context_indices: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("contextIndices", "context_indices"),
        serialization_alias="contextIndices",
    )
    input_view_count: int | None = Field(
        default=None,
        validation_alias=AliasChoices("inputViewCount", "input_view_count"),
        serialization_alias="inputViewCount",
    )
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("options", mode="before")
    @classmethod
    def validate_options(cls, value: Any) -> dict[str, Any]:
        if value is None or value == "":
            return dict(DEFAULT_TASK_OPTIONS)
        if isinstance(value, dict):
            merged = dict(DEFAULT_TASK_OPTIONS)
            merged.update(normalize_task_options(value))
            return merged
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("options JSON must decode to an object")
            merged = dict(DEFAULT_TASK_OPTIONS)
            merged.update(normalize_task_options(parsed))
            return merged
        raise ValueError("options must be a JSON object or JSON string")

    @field_validator("images", mode="before")
    @classmethod
    def validate_images(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        raise ValueError("images must be a list")

    @field_validator("video", mode="before")
    @classmethod
    def validate_video(cls, value: Any) -> str | None:
        if value is None or value == "":
            return None
        return str(value)

    @field_validator("context_indices", mode="before")
    @classmethod
    def validate_context_indices(cls, value: Any) -> list[int]:
        if value is None or value == "":
            return []
        parsed = value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = [item.strip() for item in text.split(",") if item.strip()]
        if not isinstance(parsed, list):
            raise ValueError("contextIndices must be a list")
        indices = [int(item) for item in parsed]
        if any(index < 0 for index in indices):
            raise ValueError("contextIndices values must be non-negative")
        return sorted(set(indices))

    @model_validator(mode="after")
    def validate_source(self) -> "CreateTaskRequest":
        has_sample = self.sample_id is not None
        has_images = bool(self.images)
        has_video = self.video is not None
        has_pose = self.pose_file is not None

        if has_sample:
            if has_images or has_video or self.context_indices or has_pose:
                raise ValueError("sampleId cannot be combined with images, video, poseFile, or contextIndices")
            return self

        if not has_images and not has_video:
            raise ValueError("Either sampleId or images or video is required")

        if has_images and has_video:
            raise ValueError("Provide either images or video, not both")

        if has_pose and not has_images:
            raise ValueError("poseFile requires images to be uploaded")

        return self


class CreateTaskResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    state: TaskState
    created_at: datetime = Field(serialization_alias="createdAt")


class TaskTimings(BaseModel):
    queued_at: datetime | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    duration_seconds: float | None = None
    data_load_seconds: float | None = None
    data_prep_seconds: float | None = None
    inference_seconds: float | None = None
    splat_conversion_seconds: float | None = None


class TaskResponse(BaseModel):
    task_id: str
    status: TaskState
    stage: str
    message: str | None = None
    created_at: datetime
    updated_at: datetime
    timings: TaskTimings | None = None
    input_images: list[str] = Field(default_factory=list)
    sample_id: str | None = None
    sample_name: str | None = None
    preset_id: str | None = None
    preset_name: str | None = None


class CancelTaskRequest(BaseModel):
    reason: str = "frontend_cancel_request"


class CancelTaskResponse(BaseModel):
    task_id: str
    status: TaskState
    cancel: CancelMetadata | None = None
    error_summary: str | None = None


class TaskLogEntry(BaseModel):
    id: str
    timestamp: datetime
    level: str
    message: str


class TaskLogsResponse(BaseModel):
    task_id: str
    entries: list[TaskLogEntry] = Field(default_factory=list)


class TaskResultResponse(TaskResponse):
    video_url: str | None = None
    depth_images: list[str] = Field(default_factory=list)
    splat_url: str | None = None
    rendered_images: list[str] = Field(default_factory=list)
    gt_images: list[str] = Field(default_factory=list)
    error_images: list[str] = Field(default_factory=list)
    comparison_concat_image: str | None = None
    images_zip_url: str | None = None
    scores: dict[str, Any] = Field(default_factory=dict)
    camera_params_url: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    camera_intrinsics: list[dict[str, float]] = Field(default_factory=list)
    camera_extrinsics: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str | dict[str, Any]


class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


class CameraExtrinsics(BaseModel):
    rotation: list[list[float]]
    translation: list[float]


class InputImageItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    url: str
    index: int
    width: int | None = None
    height: int | None = None
    isContextView: bool = Field(serialization_alias="isContextView")
    participatesInInference: bool = Field(serialization_alias="participatesInInference")
    cameraIntrinsics: CameraIntrinsics | None = None
    cameraExtrinsics: CameraExtrinsics | None = None
    cameraParamsStatus: Literal["available", "missing"] = Field(serialization_alias="cameraParamsStatus")


class InputImagesResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    taskId: str = Field(serialization_alias="taskId")
    images: list[InputImageItem] = Field(default_factory=list)
