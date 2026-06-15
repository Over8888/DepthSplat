from __future__ import annotations

from pathlib import Path

import numpy as np

from depth_sources.base import load_depth_results
from reproject import reproject_depth_pair


def compute_cdce_for_method(method_dir: Path) -> dict:
    """Compute cross-view depth consistency error for one method.

    For each pair of views (i, j) with i != j, reprojects depth_i to view_j
    and computes the relative error.

    Returns dict with:
      cdce: overall mean CDCE
      cdce_pairs: dict mapping (i, j) -> cdce value
    """
    data = load_depth_results(method_dir)
    depths = data["depths"]
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    V = len(depths)

    assert extrinsics.shape[0] == V
    assert intrinsics.shape[0] == V

    if extrinsics.shape[1] == 3 and extrinsics.shape[2] == 4:
        pad = np.tile(np.array([0, 0, 0, 1]), (V, 1, 1))
        extrinsics_4x4 = np.concatenate([
            extrinsics,
            np.zeros((V, 1, 4)),
        ], axis=1)
        for v in range(V):
            extrinsics_4x4[v, 3, 3] = 1.0
    elif extrinsics.shape[1:] == (4, 4):
        extrinsics_4x4 = extrinsics
    else:
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")

    cdce_pairs = {}
    pair_results = {}

    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            result = reproject_depth_pair(
                depths[i],
                extrinsics_4x4[i],
                intrinsics[i],
                depths[j],
                extrinsics_4x4[j],
                intrinsics[j],
            )
            cdce = np.nanmean(result["error_map"])
            cdce_pairs[(i, j)] = cdce
            pair_results[(i, j)] = result

    overall_cdce = np.nanmean(list(cdce_pairs.values()))

    return {
        "cdce": overall_cdce,
        "cdce_pairs": cdce_pairs,
        "pair_results": pair_results,
    }


def compute_cdce_all(scene_dir: Path, methods: list[str]) -> dict:
    results = {}
    for method in methods:
        method_dir = scene_dir / method
        if not method_dir.exists():
            print(f"  [WARN] {method} dir not found: {method_dir}")
            continue
        try:
            method_result = compute_cdce_for_method(method_dir)
            results[method] = method_result
        except Exception as e:
            print(f"  [ERROR] {method}: {e}")
            import traceback
            traceback.print_exc()
    return results
