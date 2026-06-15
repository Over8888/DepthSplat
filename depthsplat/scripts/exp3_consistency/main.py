#!/usr/bin/env python
"""Experiment 3: Multi-view Depth Consistency.

Compares cross-view depth consistency across 4 methods:
  Ours (DepthSplat + PMR), MVSplat, VGGT, DA3-BASE

Usage:
  python main.py                    # Run all 5 scenes
  python main.py --scene SCENE_ID   # Run a single scene
  python main.py --skip-inference   # Only compute metrics (inference already done)
  python main.py --skip-metrics     # Only run inference (no CDCE calc)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from depth_sources.depthsplat import DepthSplatSource
from depth_sources.mvsplat_source import MVSplatSource
from depth_sources.vggt_source import VGGTSource, DA3Source
from metrics import compute_cdce_for_method, compute_cdce_all
from utils import load_scene_data, save_scene_data
from visualize import (
    generate_comparison_figure,
    generate_single_method_reproj_visualization,
    save_cdce_bar_chart,
    save_cdce_table,
)


def get_sources() -> dict:
    return {
        "ours": DepthSplatSource(),
        "mvsplat": MVSplatSource(),
        "vggt": VGGTSource(),
        "da3": DA3Source(),
    }


def run_scene_inference(scene_id: str, sources: dict, force: bool = False) -> bool:
    scene_output_dir = config.get_scene_output_dir(scene_id)
    all_exist = all(
        (scene_output_dir / m / "cameras.npz").exists()
        for m in config.METHODS
    )
    if all_exist and not force:
        print(f"[skip] Scene {scene_id} already has all inference outputs")
        return True

    print(f"\n{'='*60}")
    print(f"Loading scene data: {scene_id}")
    print(f"{'='*60}")

    try:
        data = load_scene_data(scene_id, config.DATASET_ROOT)
    except Exception as e:
        print(f"  [ERROR] Failed to load scene {scene_id}: {e}")
        return False

    scene_images_dir = scene_output_dir / "input_images"
    image_paths = save_scene_data(data, scene_images_dir)

    data["scene"] = scene_id
    print(f"  Context views: {data['indices']}")
    print(f"  Image shape: {data['images'].shape}")

    for method_name in config.METHODS:
        if method_name not in sources:
            continue
        method_dir = config.get_method_dir(scene_id, method_name)
        if (method_dir / "cameras.npz").exists() and not force:
            print(f"  [{method_name}] Already done, skipping")
            continue

        print(f"\n  [{method_name}] Running inference...")
        t0 = time.time()
        try:
            sources[method_name].predict(image_paths, data, method_dir)
            elapsed = time.time() - t0
            print(f"  [{method_name}] Done in {elapsed:.1f}s")
        except Exception as e:
            print(f"  [{method_name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def run_scene_cdce(scene_id: str, sources: dict) -> dict | None:
    scene_output_dir = config.get_scene_output_dir(scene_id)
    print(f"\n  Computing CDCE for scene {scene_id}...")

    try:
        results = compute_cdce_all(scene_output_dir, config.METHODS)
    except Exception as e:
        print(f"  [ERROR] CDCE computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    for method, result in sorted(results.items()):
        pair_results = result.get("pair_results", {})
        display = sources.get(method)
        display_name = display.display_name if display else method
        if pair_results:
            method_dir = config.get_method_dir(scene_id, method)
            generate_single_method_reproj_visualization(
                method_dir, pair_results, (0, 1), config.METHODS
            )
        print(f"    {display_name}: CDCE = {result['cdce']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Multi-view Depth Consistency")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scenes = [args.scene] if args.scene else config.SCENE_IDS
    sources = get_sources()

    print("=" * 60)
    print("Experiment 3: Multi-view Depth Consistency")
    print(f"Methods: {', '.join(s.display_name for s in sources.values())}")
    print(f"Scenes: {scenes}")
    print("=" * 60)

    all_cdce_results = {}

    for scene_id in scenes:
        print(f"\n{'#'*60}")
        print(f"# Scene: {scene_id}")
        print(f"{'#'*60}")

        if not args.skip_inference:
            success = run_scene_inference(scene_id, sources, force=args.force)
            if not success:
                print(f"  [SKIP] Scene {scene_id} inference failed, skipping CDCE")
                continue

        if not args.skip_metrics:
            cdce_result = run_scene_cdce(scene_id, sources)
            if cdce_result is not None:
                all_cdce_results[scene_id] = cdce_result

    if all_cdce_results and not args.skip_metrics:
        print(f"\n{'='*60}")
        print("Generating final figures and tables")
        print(f"{'='*60}")

        figures_dir = config.OUTPUT_ROOT / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        save_cdce_table(all_cdce_results, figures_dir / "cdce_results.csv", config.METHODS)
        save_cdce_bar_chart(all_cdce_results, figures_dir / "cdce_bar_chart.png", config.METHODS)

        for scene_id in all_cdce_results:
            generate_comparison_figure(
                config.get_scene_output_dir(scene_id),
                {scene_id: all_cdce_results[scene_id]},
                figures_dir / f"comparison_{scene_id}.png",
                config.METHODS,
                pair=(0, 1),
            )

        best_method = None
        best_cdce = float("inf")
        for method in config.METHODS:
            vals = []
            for s, r in all_cdce_results.items():
                if method in r:
                    vals.append(r[method]["cdce"])
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {sources[method].display_name}: avg CDCE = {avg:.6f}")
                if avg < best_cdce:
                    best_cdce = avg
                    best_method = method

        if best_method:
            print(f"\n  Best method: {sources[best_method].display_name} (CDCE={best_cdce:.6f})")

    print(f"\nAll outputs saved to: {config.OUTPUT_ROOT}")
    print("Done.")


if __name__ == "__main__":
    main()
