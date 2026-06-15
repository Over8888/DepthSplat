from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from app.config import Settings
from app.services.vggt_service import CameraPrediction
from app.utils.logging import get_logger


class TTT3RService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)

    def predict_cameras(self, image_paths: list[Path]) -> CameraPrediction:
        if not image_paths:
            raise ValueError("At least one image is required for TTT3R camera estimation")
        prediction, meta = self._run_subprocess(image_paths)
        self.logger.info(
            "ttt3r prediction completed",
            event="ttt3r_prediction_completed",
            fields={
                "image_count": len(image_paths),
                "device": meta.get("device"),
                "checkpoint": meta.get("source"),
                "execution_mode": "subprocess",
                "model_update_type": meta.get("model_update_type"),
                "reset_interval": meta.get("reset_interval"),
            },
        )
        return prediction

    def _build_request_payload(self, image_paths: list[Path]) -> dict:
        return {
            "image_paths": [str(path) for path in image_paths],
            "ttt3r_root": str(self.settings.ttt3r_root),
            "model_path": str(self.settings.ttt3r_model_path),
            "size": self.settings.ttt3r_size,
            "model_update_type": self.settings.ttt3r_model_update_type,
            "reset_interval": self.settings.ttt3r_reset_interval,
            "device": self.settings.ttt3r_device,
        }

    def _subprocess_command(self, request_path: Path, output_path: Path, meta_output_path: Path) -> list[str]:
        return [
            self.settings.ttt3r_python,
            "-m",
            "app.services.ttt3r_subprocess",
            "--request",
            str(request_path),
            "--output",
            str(output_path),
            "--meta-output",
            str(meta_output_path),
        ]

    def _run_subprocess(self, image_paths: list[Path]) -> tuple[CameraPrediction, dict]:
        with tempfile.TemporaryDirectory(prefix="depthsplat_v3_ttt3r_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            request_path = tmpdir_path / "request.json"
            output_path = tmpdir_path / "result.npz"
            meta_output_path = tmpdir_path / "result_meta.json"
            request_path.write_text(
                json.dumps(self._build_request_payload(image_paths), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            command = self._subprocess_command(request_path, output_path, meta_output_path)
            self.logger.info(
                "ttt3r subprocess starting",
                event="ttt3r_subprocess_starting",
                fields={"image_count": len(image_paths), "command": command},
            )
            result = subprocess.run(
                command,
                cwd=str(self.settings.app_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "TTT3R subprocess failed"
                    + self._format_subprocess_output(result.stdout, result.stderr, result.returncode)
                )
            if not output_path.exists():
                raise RuntimeError("TTT3R subprocess completed without writing result.npz")
            if not meta_output_path.exists():
                raise RuntimeError("TTT3R subprocess completed without writing result metadata")

            with np.load(output_path) as payload:
                prediction = CameraPrediction(
                    extrinsics_w2c=payload["extrinsics_w2c"],
                    intrinsics_px=payload["intrinsics_px"],
                )
            meta = json.loads(meta_output_path.read_text(encoding="utf-8"))
            return prediction, meta

    @staticmethod
    def _format_subprocess_output(stdout: str, stderr: str, return_code: int) -> str:
        chunks = [f" (exit code {return_code})"]
        if stdout.strip():
            chunks.append(f"\n[stdout]\n{stdout.strip()}")
        if stderr.strip():
            chunks.append(f"\n[stderr]\n{stderr.strip()}")
        return "".join(chunks)
