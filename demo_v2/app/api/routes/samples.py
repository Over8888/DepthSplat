from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.schemas.samples import PresetItem, SampleItem, SamplesResponse

router = APIRouter(tags=["samples"])


@router.get("/samples", response_model=SamplesResponse)
def get_samples(request: Request, preset: str | None = Query(None)) -> SamplesResponse:
    manager = request.app.state.task_manager
    preset_name = preset or manager.settings.default_preset
    items = manager.list_samples(preset_name)
    preset_items = []
    for preset_cfg in manager.list_presets():
        preset_items.append(
            PresetItem(
                name=preset_cfg.name,
                display_name=preset_cfg.display_name,
                defaults={
                    "checkpoint": preset_cfg.checkpoint_path.name,
                    "num_context_views": preset_cfg.num_context_views,
                    "image_shape": list(preset_cfg.image_shape),
                },
                sample_count=len(manager.list_samples(preset_cfg.name)),
            )
        )
    return SamplesResponse(
        default_preset=manager.settings.default_preset,
        presets=preset_items,
        items=[
            SampleItem(id=item.id, preset=item.preset, scene_key=item.scene_key, label=item.label, defaults=item.defaults)
            for item in items
        ],
    )
