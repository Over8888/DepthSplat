from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class DepthSource(ABC):
    """Abstract interface for depth prediction methods."""

    @abstractmethod
    def predict(
        self,
        image_paths: list[Path],
        cameras: dict[str, np.ndarray],
        output_dir: Path,
    ) -> None:
        """Run depth inference and save results to output_dir.

        Must create:
          output_dir/depth_%06d.npy  -- metric depth [H, W] float32
          output_dir/images_%06d.png -- RGB images at inference resolution
          output_dir/cameras.npz     -- dict with extrinsics, intrinsics, near, far
        """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def display_name(self) -> str:
        return self.name


def load_depth_results(method_dir: Path) -> dict[str, Any]:
    """Load saved depth results from a method directory."""
    import glob

    cameras = dict(np.load(method_dir / "cameras.npz"))
    depth_files = sorted(method_dir.glob("depth_*.npy"))
    depths = np.stack([np.load(f) for f in depth_files], axis=0)

    return {
        "depths": depths,
        "extrinsics": cameras["extrinsics"],
        "intrinsics": cameras["intrinsics"],
        "near": float(cameras.get("near", 1.0)),
        "far": float(cameras.get("far", 100.0)),
    }
