from __future__ import annotations

from pathlib import Path

import numpy as np

from depth_sources.base import DepthSource
import config
from utils import run_subprocess


class VGGTSource(DepthSource):
    def __init__(self):
        self._name = "vggt"

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return "VGGT"

    def predict(
        self,
        image_paths: list[Path],
        cameras: dict[str, np.ndarray],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        infer_script = Path(__file__).parent / "vggt_infer.py"
        cmd = [
            config.PYTHON_BIN, str(infer_script),
            "--image-dir", str(image_paths[0].parent),
            "--output-dir", str(output_dir),
        ]

        ret, stdout, stderr = run_subprocess(
            cmd, cwd=config.PROJECT_ROOT,
            env={"CUDA_VISIBLE_DEVICES": "0"},
        )
        print(f"[VGGT stdout]\n{stdout}")
        if stderr:
            print(f"[VGGT stderr]\n{stderr}")
        if ret != 0:
            raise RuntimeError(f"VGGT failed (exit {ret})\n{stderr}")

        import shutil
        for idx, img_path in enumerate(image_paths):
            shutil.copy(img_path, output_dir / f"image_{idx:06d}.png")


class DA3Source(DepthSource):
    def __init__(self):
        self._name = "da3"

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return "DA3-BASE"

    def predict(
        self,
        image_paths: list[Path],
        cameras: dict[str, np.ndarray],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        infer_script = Path(__file__).parent / "da3_infer.py"
        cmd = [
            config.PYTHON_BIN, str(infer_script),
            "--image-dir", str(image_paths[0].parent),
            "--output-dir", str(output_dir),
        ]

        ret, stdout, stderr = run_subprocess(
            cmd, cwd=config.PROJECT_ROOT,
            env={"CUDA_VISIBLE_DEVICES": "0"},
        )
        print(f"[DA3 stdout]\n{stdout}")
        if stderr:
            print(f"[DA3 stderr]\n{stderr}")
        if ret != 0:
            raise RuntimeError(f"DA3 failed (exit {ret})\n{stderr}")

        import shutil
        for idx, img_path in enumerate(image_paths):
            shutil.copy(img_path, output_dir / f"image_{idx:06d}.png")
