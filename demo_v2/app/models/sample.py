from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class SampleRecord(BaseModel):
    id: str
    preset: str
    scene_key: str
    label: str
    source_chunk: Path
    defaults: dict
