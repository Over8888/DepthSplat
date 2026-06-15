from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SampleRecord(BaseModel):
    id: str
    preset: str
    scene_key: str
    label: str
    source_chunk: Path
    defaults: dict
    name: str | None = None
    description: str | None = None
    thumbnail_url: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    input_images: list[str] = Field(default_factory=list)
    preview_images: list[str] = Field(default_factory=list)
    scene_number: str | int | None = None
    input_view_count: int | None = None
    target_view_count: int | None = None
