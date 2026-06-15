from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image


def _load_image(path: Path, target_h: int = 256, target_w: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img)


def _depth_to_color(depth: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    valid = ~np.isnan(depth) & (depth > 0)
    if vmin is None:
        vmin = np.nanpercentile(depth[valid], 5) if valid.any() else 0
    if vmax is None:
        vmax = np.nanpercentile(depth[valid], 95) if valid.any() else 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    colored = cm.plasma(norm(depth))
    colored[~valid] = [0, 0, 0, 1]
    return (colored[:, :, :3] * 255).astype(np.uint8)


def _error_to_color(error_map: np.ndarray, vmax: float = 0.3) -> np.ndarray:
    valid = ~np.isnan(error_map) & (error_map >= 0)
    norm = Normalize(vmin=0, vmax=vmax)
    colored = cm.hot(norm(np.clip(error_map, 0, vmax)))
    colored[~valid] = [0, 0, 0, 1]
    return (colored[:, :, :3] * 255).astype(np.uint8)


def generate_comparison_figure(
    scene_dir: Path,
    cdce_results: dict,
    output_path: Path,
    methods: list[str],
    pair: tuple[int, int] = (0, 1),
) -> None:
    src_idx, tgt_idx = pair
    method_dirs = {m: scene_dir / m for m in methods if (scene_dir / m).exists()}

    if not method_dirs:
        return

    num_methods = len(method_dirs)
    num_cols = 4
    fig, axes = plt.subplots(
        num_methods, num_cols,
        figsize=(num_cols * 4, num_methods * 4.5),
        gridspec_kw={"wspace": 0.05, "hspace": 0.15},
    )
    if num_methods == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        f"View {src_idx} RGB",
        f"View {tgt_idx} RGB",
        f"Reprojected {src_idx}->{tgt_idx}",
        f"Error Map {src_idx}->{tgt_idx}",
    ]

    ref_vmin, ref_vmax = None, None
    for row_idx, (method, mdir) in enumerate(sorted(method_dirs.items())):
        img_src = _load_image(mdir / f"image_{src_idx:06d}.png")
        img_tgt = _load_image(mdir / f"image_{tgt_idx:06d}.png")

        axes[row_idx, 0].imshow(img_src)
        axes[row_idx, 1].imshow(img_tgt)

        depth_reproj_file = mdir / f"reproj_{src_idx}_to_{tgt_idx}_reproj.npy"
        error_file = mdir / f"reproj_{src_idx}_to_{tgt_idx}_error.npy"

        if depth_reproj_file.exists():
            depth_reproj = np.load(depth_reproj_file)
        else:
            depth_reproj = np.zeros((256, 256))

        if error_file.exists():
            error_map = np.load(error_file)
        else:
            error_map = np.zeros((256, 256))

        valid_reproj = ~np.isnan(depth_reproj) & (depth_reproj > 0)
        if valid_reproj.any() and ref_vmin is None:
            ref_vmin = np.nanpercentile(depth_reproj[valid_reproj], 5)
            ref_vmax = np.nanpercentile(depth_reproj[valid_reproj], 95)

        rviz = _depth_to_color(depth_reproj, vmin=ref_vmin, vmax=ref_vmax)
        eviz = _error_to_color(error_map)

        axes[row_idx, 2].imshow(rviz)
        axes[row_idx, 3].imshow(eviz)

        axes[row_idx, 0].set_ylabel(
            _lookup_display_name(method), fontsize=12, fontweight="bold"
        )

    for col in range(num_cols):
        axes[0, col].set_title(col_titles[col], fontsize=11)
        for row in range(num_methods):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved comparison figure to {output_path}")


def _lookup_display_name(method: str) -> str:
    names = {
        "ours": "Ours (PMR)",
        "mvsplat": "MVSplat",
        "vggt": "VGGT",
        "da3": "DA3-BASE",
    }
    return names.get(method, method)


def generate_single_method_reproj_visualization(
    method_dir: Path, pair_results: dict, pair: tuple[int, int], methods: list[str]
) -> None:
    for (i, j), result in pair_results.items():
        method_dir_i = method_dir
        if not method_dir_i.exists():
            method_dir_i.mkdir(parents=True, exist_ok=True)
        np.save(method_dir / f"reproj_{i}_to_{j}_reproj.npy", result["depth_reproj"])
        np.save(method_dir / f"reproj_{i}_to_{j}_error.npy", result["error_map"])
        np.save(method_dir / f"reproj_{i}_to_{j}_valid.npy", result["valid"])


def save_cdce_table(
    cdce_results: dict[str, dict], output_path: Path, methods: list[str]
) -> None:
    scenes = sorted(cdce_results.keys())
    rows = []
    for scene in scenes:
        row = {"Scene": scene}
        for method in methods:
            if method in cdce_results[scene]:
                row[method] = f"{cdce_results[scene][method]['cdce']:.6f}"
            else:
                row[method] = "N/A"
        rows.append(row)

    avg_row = {"Scene": "Average"}
    for method in methods:
        values = []
        for row in rows:
            if row[method] != "N/A":
                values.append(float(row[method]))
        avg_row[method] = f"{np.nanmean(values):.6f}" if values else "N/A"
    rows.append(avg_row)

    fieldnames = ["Scene"] + methods
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CDCE table to {output_path}")

    json_data = {}
    for scene in scenes:
        json_data[scene] = {}
        for method in methods:
            if method in cdce_results[scene]:
                json_data[scene][method] = {
                    "cdce": float(cdce_results[scene][method]["cdce"]),
                    "cdce_pairs": {
                        f"{i}->{j}": float(v)
                        for (i, j), v in cdce_results[scene][method]["cdce_pairs"].items()
                    },
                }
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved CDCE JSON to {json_path}")


def save_cdce_bar_chart(
    cdce_results: dict[str, dict], output_path: Path, methods: list[str]
) -> None:
    scenes = sorted(cdce_results.keys())
    x = np.arange(len(scenes))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))

    for mi, method in enumerate(methods):
        values = []
        for scene in scenes:
            if method in cdce_results[scene]:
                values.append(cdce_results[scene][method]["cdce"])
            else:
                values.append(np.nan)
        bars = ax.bar(
            x + mi * width - width * (len(methods) - 1) / 2,
            values,
            width,
            label=_lookup_display_name(method),
        )

    ax.set_ylabel("CDCE")
    ax.set_xticks(x)
    ax.set_xticklabels([s[:8] for s in scenes], rotation=30, ha="right")
    ax.legend()
    ax.set_title("Cross-view Depth Consistency Error (lower is better)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved bar chart to {output_path}")
