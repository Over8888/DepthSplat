from __future__ import annotations

import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import PresetConfig, Settings
from app.models.enums import TaskState
from app.models.task import ProcessInfo
from app.schemas.tasks import CreateTaskRequest
from app.services.result_builder import ResultBuilder
from app.services.sample_service import SampleService
from app.services.storage import FilesystemStorage
from app.services.task_manager import TaskManager


class FakeRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._active: dict[str, threading.Event] = {}

    def build_command(self, preset, dataset_root, eval_index_path, output_dir):
        return ["fake-depthsplat", preset.name, str(dataset_root), str(eval_index_path), str(output_dir)]

    def write_command_files(self, task_dir, preset, command):
        logs = task_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "cmd.txt").write_text(" ".join(command), encoding="utf-8")
        (logs / "cmd.sh").write_text("#!/usr/bin/env bash\n" + " ".join(command) + "\n", encoding="utf-8")

    def start_and_wait(self, task_id, preset, command, task_dir):
        stop = threading.Event()
        self._active[task_id] = stop
        started_at = datetime.now(timezone.utc)
        stdout = task_dir / "logs" / "stdout.log"
        stderr = task_dir / "logs" / "stderr.log"
        stdout.write_text("fake run start\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        output_dir = task_dir / "meta" / "depthsplat_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(20):
            if stop.is_set():
                finished = datetime.now(timezone.utc)
                return type("RunnerResult", (), {"return_code": -15, "started_at": started_at, "finished_at": finished, "runtime_seconds": 0.5, "cancelled": True})
            time.sleep(0.05)
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)
        (output_dir / "videos" / "fake.mp4").write_text("fake", encoding="utf-8")
        depth_root = output_dir / "images" / "fake_scene" / "depth"
        depth_root.mkdir(parents=True, exist_ok=True)
        (depth_root / "000000.png").write_text("fake", encoding="utf-8")
        (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics" / "benchmark.json").write_text('{"encoder": [0.1], "decoder": [0.2]}', encoding='utf-8')
        finished = datetime.now(timezone.utc)
        return type("RunnerResult", (), {"return_code": 0, "started_at": started_at, "finished_at": finished, "runtime_seconds": 1.0, "cancelled": False})

    def cancel(self, task_id, grace_seconds):
        evt = self._active.get(task_id)
        if not evt:
            return False, False, "No active subprocess found"
        evt.set()
        return True, False, None


class FakeSampleService(SampleService):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def list_presets(self):
        return list(self.settings.presets.values())

    def list_samples(self, preset_name: str):
        return [type("Sample", (), {"id": f"{preset_name}:fake_scene", "preset": preset_name, "scene_key": "fake_scene", "label": "fake_scene", "defaults": {}})]

    def get_sample(self, sample_id: str):
        preset_name, _ = sample_id.split(":", 1)
        return type("Sample", (), {"id": sample_id, "preset": preset_name, "scene_key": "fake_scene", "label": "fake_scene", "defaults": {}})

    def materialize_sample(self, task_dir: Path, sample_id: str, preset_name: str):
        (task_dir / "input" / "preview").mkdir(parents=True, exist_ok=True)
        (task_dir / "input" / "preview" / "context_00_000000.png").write_text("fake", encoding="utf-8")
        dataset_root = task_dir / "input" / "dataset"
        dataset_root.mkdir(parents=True, exist_ok=True)
        eval_index_path = task_dir / "meta" / "evaluation_index.json"
        eval_index_path.write_text('{"fake_scene":{"context":[0],"target":[0]}}', encoding='utf-8')
        return {
            "scene_key": "fake_scene",
            "dataset_root": dataset_root,
            "evaluation_index_path": eval_index_path,
            "input_preview_files": ["input/preview/context_00_000000.png"],
            "context_indices": [0],
            "target_count": 1,
        }


def build_fake_settings(tmp_root: Path) -> Settings:
    preset = PresetConfig(
        name="fake",
        display_name="Fake",
        experiment="fake",
        dataset_root=tmp_root,
        checkpoint_path=tmp_root / "fake.pth",
        fixed_index_path=tmp_root / "fake.json",
        num_context_views=1,
        image_shape=(1, 1),
    )
    return Settings(
        app_root=tmp_root,
        outputs_root=tmp_root / "outputs" / "tasks",
        depthsplat_root=tmp_root,
        depthsplat_python="python",
        host="127.0.0.1",
        port=8012,
        default_preset="fake",
        cancellation_grace_seconds=1.0,
        log_tail_lines=50,
        presets={"fake": preset},
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        settings = build_fake_settings(Path(tmp))
        storage = FilesystemStorage(settings)
        sample_service = FakeSampleService(settings)
        runner = FakeRunner(settings)
        result_builder = ResultBuilder(settings, storage)
        manager = TaskManager(settings, storage, sample_service, runner, result_builder)
        manager.start()

        task = manager.create_task(CreateTaskRequest(sample_id="fake:fake_scene", preset="fake"))
        time.sleep(2)
        task = storage.load_task(task.id)
        assert task.status == TaskState.success, task.status
        for _ in range(20):
            if (storage.task_dir(task.id) / "logs" / "stdout.log").exists() and storage.meta_path(task.id, "result.json").exists():
                break
            time.sleep(0.1)
        assert (storage.task_dir(task.id) / "logs" / "stdout.log").exists()
        assert storage.meta_path(task.id, "result.json").exists()

        cancel_task = manager.create_task(CreateTaskRequest(sample_id="fake:fake_scene", preset="fake"))
        time.sleep(0.2)
        manager.cancel_task(cancel_task.id, "validation_cancel")
        time.sleep(1)
        cancel_task = storage.load_task(cancel_task.id)
        assert cancel_task.status == TaskState.cancelled, cancel_task.status
        assert storage.meta_path(cancel_task.id, "cancel.json").exists()
        print("validation ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
