from __future__ import annotations

import json
import sys
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings
from app.utils.logging import get_logger


@dataclass(frozen=True)
class CameraPrediction:
    extrinsics_w2c: np.ndarray
    intrinsics_px: np.ndarray


class VGGTService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)

    def predict_cameras(self, image_paths: list[Path]) -> CameraPrediction:
        if not image_paths:
            raise ValueError("At least one image is required for VGGT camera estimation")
        prediction, meta = self._run_subprocess(image_paths)
        self.logger.info(
            "vggt prediction completed",
            event="vggt_prediction_completed",
            fields={
                "image_count": len(image_paths),
                "device": meta.get("device"),
                "checkpoint": meta.get("source"),
                "execution_mode": "subprocess",
            },
        )
        return prediction

    def _resolve_checkpoint_path(self) -> Path | None:
        if self.settings.vggt_checkpoint_path is not None:
            explicit = self.settings.vggt_checkpoint_path.expanduser()
            if explicit.exists():
                return explicit
            raise RuntimeError(f"Configured VGGT checkpoint not found: {explicit}")

        direct_candidates = [
            self.settings.vggt_root / "model.pt",
            self.settings.vggt_root / "checkpoints" / "model.pt",
            self.settings.vggt_root / "weights" / "model.pt",
        ]
        for candidate in direct_candidates:
            if candidate.exists():
                return candidate

        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        for glob_pattern in (
            "models--facebook--VGGT-1B*/snapshots/*/model.pt",
            "models--facebook--VGGT-1B-Commercial*/snapshots/*/model.pt",
        ):
            for candidate in sorted(cache_root.glob(glob_pattern), reverse=True):
                if candidate.exists():
                    return candidate

        return None

    def _build_request_payload(self, image_paths: list[Path]) -> dict:
        checkpoint_path = self._resolve_checkpoint_path()
        return {
            "image_paths": [str(path) for path in image_paths],
            "vggt_root": str(self.settings.vggt_root),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "model_id": self.settings.vggt_model_id,
            "load_resolution": self.settings.vggt_load_resolution,
            "inference_resolution": self.settings.vggt_inference_resolution,
            "device": self.settings.vggt_device,
        }

    def _subprocess_command(self, request_path: Path, output_path: Path, meta_output_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "app.services.vggt_subprocess",
            "--request",
            str(request_path),
            "--output",
            str(output_path),
            "--meta-output",
            str(meta_output_path),
        ]

    def _run_subprocess(self, image_paths: list[Path]) -> tuple[CameraPrediction, dict]:
        with tempfile.TemporaryDirectory(prefix="depthsplat_v3_vggt_") as tmpdir:
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
                "vggt subprocess starting",
                event="vggt_subprocess_starting",
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
                    "VGGT subprocess failed"
                    + self._format_subprocess_output(result.stdout, result.stderr, result.returncode)
                )
            if not output_path.exists():
                raise RuntimeError("VGGT subprocess completed without writing result.npz")
            if not meta_output_path.exists():
                raise RuntimeError("VGGT subprocess completed without writing result metadata")

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
