from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.samples import FrontendPresetItem, PresetsResponse

router = APIRouter(tags=["presets"])


@router.get("/presets", response_model=PresetsResponse)
def get_presets(request: Request) -> PresetsResponse:
    manager = request.app.state.task_manager
    items: list[FrontendPresetItem] = []
    for preset_cfg in manager.list_presets():
        samples = manager.list_samples(preset_cfg.name)
        items.append(
            FrontendPresetItem(
                id=preset_cfg.name,
                name=preset_cfg.display_name,
                description="",
                checkpoint=str(preset_cfg.checkpoint_path.relative_to(manager.settings.depthsplat_root)),
                contextViews=preset_cfg.num_context_views,
                imageShape=list(preset_cfg.image_shape),
                sampleId=samples[0].id if samples else None,
            )
        )
    return PresetsResponse(items=items)
