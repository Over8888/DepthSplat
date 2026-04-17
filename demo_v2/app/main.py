from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.presets import router as presets_router
from app.api.routes.samples import router as samples_router
from app.api.routes.tasks import router as tasks_router
from app.config import get_settings
from app.services.result_builder import ResultBuilder
from app.services.runner import Runner
from app.services.sample_service import SampleService
from app.services.storage import FilesystemStorage
from app.services.task_manager import TaskManager
from app.utils.logging import get_logger, setup_logging


def create_app() -> FastAPI:
    setup_logging()
    logger = get_logger(__name__)
    settings = get_settings()
    storage = FilesystemStorage(settings)
    sample_service = SampleService(settings)
    runner = Runner(settings)
    result_builder = ResultBuilder(settings, storage)
    task_manager = TaskManager(settings, storage, sample_service, runner, result_builder)
    task_manager.start()

    app = FastAPI(title="DepthSplat v3 Linux Backend", version="3.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_origin_regex=r".*",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.storage = storage
    app.state.task_manager = task_manager

    app.include_router(presets_router)
    app.include_router(samples_router)
    app.include_router(tasks_router)
    app.mount("/artifacts", StaticFiles(directory=str(settings.outputs_root)), name="artifacts")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    logger.info("app created", event="app_created", fields={"outputs_root": str(settings.outputs_root)})
    return app


app = create_app()
