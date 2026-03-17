from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from app.models.enums import TaskState
from app.models.task import CancelMetadata


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sample_id: str = Field(validation_alias=AliasChoices("sampleId", "sample_id"), serialization_alias="sampleId")
    preset_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("presetId", "preset"),
        serialization_alias="presetId",
    )
    images: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("options", mode="before")
    @classmethod
    def validate_options(cls, value: Any) -> dict[str, Any]:
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("options JSON must decode to an object")
            return parsed
        raise ValueError("options must be a JSON object or JSON string")

    @field_validator("images", mode="before")
    @classmethod
    def validate_images(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        raise ValueError("images must be a list")


class CreateTaskResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    state: TaskState
    created_at: datetime = Field(serialization_alias="createdAt")


class TaskResponse(BaseModel):
    task_id: str
    status: TaskState
    stage: str
    created_at: datetime
    updated_at: datetime
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


class TaskLogsResponse(BaseModel):
    task_id: str
    status: TaskState
    stage: str
    stdout: str = ""
    stderr: str = ""


class TaskResultResponse(TaskResponse):
    video_url: str | None = None


class ErrorResponse(BaseModel):
    detail: str | dict[str, Any]
