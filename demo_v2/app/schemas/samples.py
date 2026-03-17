from __future__ import annotations

from pydantic import BaseModel, Field


class SampleItem(BaseModel):
    id: str
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
