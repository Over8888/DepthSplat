from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class SampleItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
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
    preset: str
    scene_key: str
    label: str
    defaults: dict = Field(default_factory=dict)


class PresetItem(BaseModel):
    name: str
    display_name: str
    defaults: dict = Field(default_factory=dict)
    sample_count: int


class SamplesResponse(BaseModel):
    default_preset: str
    presets: list[PresetItem]
    items: list[SampleItem]


class FrontendPresetItem(BaseModel):
    id: str
    name: str
    description: str = ""
    checkpoint: str
    contextViews: int
    imageShape: list[int]
    sampleId: str | None = None


class PresetsResponse(BaseModel):
    items: list[FrontendPresetItem]
