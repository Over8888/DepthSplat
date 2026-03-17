from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .enums import TaskState


class TimingInfo(BaseModel):
    queued_at: datetime | None = None
    preparing_started_at: datetime | None = None
    running_started_at: datetime | None = None
    postprocessing_started_at: datetime | None = None
    finished_at: datetime | None = None
    startup_seconds: float | None = None
    model_load_seconds: float | None = None
    data_prep_seconds: float | None = None
    forward_seconds: float | None = None
    save_outputs_seconds: float | None = None
    total_seconds: float | None = None


class CancelMetadata(BaseModel):
    requested_at: datetime
    finalised_at: datetime | None = None
    reason: str
    requested_by: str = "frontend"
    kill_sent: bool = False
    force_kill_sent: bool = False
    outcome: str | None = None
    error: str | None = None


class RequestMetadata(BaseModel):
    preset: str
    sample_id: str
    scene_key: str
    checkpoint: str
    image_shape: list[int]
    num_context_views: int
    images: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)


class ResultMetadata(BaseModel):
    task_id: str
    status: TaskState
    result_complete: bool = False
    input_images: list[str] = Field(default_factory=list)
    video_url: str | None = None
    video_path: str | None = None
    depth_images: list[str] = Field(default_factory=list)
    preview_images: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error_summary: str | None = None
    cancel: CancelMetadata | None = None


class ProcessInfo(BaseModel):
    pid: int | None = None
    pgid: int | None = None
    command: list[str] = Field(default_factory=list)


class TaskRecord(BaseModel):
    id: str
    status: TaskState
    stage: TaskState
    created_at: datetime
    updated_at: datetime
    request: RequestMetadata
    timing: TimingInfo = Field(default_factory=TimingInfo)
    result: ResultMetadata | None = None
    error_summary: str | None = None
    cancel: CancelMetadata | None = None
    process: ProcessInfo = Field(default_factory=ProcessInfo)
