import csv
import json
import sys
from pathlib import Path


SKIP_DIRS = {"metrics", "videos", "gaussians", "images", "report", "report_formal"}
FORMAL_PREFIX = "formal_"
BASELINE_EXPERIMENT = "re10k_2view_baseline"


def parse_setting(path: Path) -> tuple[str, str]:
    name = path.name
    if name.startswith("formal_near_"):
        return "near_far", name.split("formal_", 1)[1]
    return "unknown", name


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_baseline_rows(
    root: Path,
    summary_rows: list[dict],
    per_scene_rows: list[dict],
) -> None:
    metrics_dir = root / BASELINE_EXPERIMENT / "metrics"
    score_path = metrics_dir / "scores_all_avg.json"
    scene_path = metrics_dir / "scores_per_scene.json"
    if not score_path.exists():
        print(f"[WARN] Missing reused baseline scores file: {score_path}")
        return

    scores = load_json(score_path)
    encoder_time = None
    decoder_time = None
    if "encoder" in scores and isinstance(scores["encoder"], list):
        encoder_time = scores["encoder"][1]
    if "decoder" in scores and isinstance(scores["decoder"], list):
        decoder_time = scores["decoder"][1]

    baseline_summary = {
        "experiment_name": BASELINE_EXPERIMENT,
        "experiment_group": "baseline",
        "setting": "num_depth_candidates=128,near=0.5,far=100",
        "output_dir": str(root / BASELINE_EXPERIMENT),
        "metrics_dir": str(metrics_dir),
        "psnr": scores.get("psnr"),
        "ssim": scores.get("ssim"),
        "lpips": scores.get("lpips"),
        "encoder_time": encoder_time,
        "decoder_time": decoder_time,
    }
    summary_rows.append(baseline_summary)

    if scene_path.exists():
        for row in load_json(scene_path):
            per_scene_rows.append(
                {
                    "experiment_name": BASELINE_EXPERIMENT,
                    "experiment_group": "baseline",
                    "setting": "num_depth_candidates=128,near=0.5,far=100",
                    "scene": row["scene"],
                    "psnr": row["psnr"],
                    "ssim": row["ssim"],
                    "lpips": row["lpips"],
                }
            )


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/summarize_formal_ablation_results.py <outputs_root>")
        return 1

    root = Path(sys.argv[1]).resolve()
    summary_rows = []
    per_scene_rows = []

    append_baseline_rows(root, summary_rows, per_scene_rows)

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS or not child.name.startswith(FORMAL_PREFIX):
            continue

        metrics_dir = child / "metrics"
        score_path = metrics_dir / "scores_all_avg.json"
        scene_path = metrics_dir / "scores_per_scene.json"
        if not score_path.exists():
            print(f"[WARN] Missing scores file: {score_path}")
            continue

        scores = load_json(score_path)
        group, setting = parse_setting(child)
        encoder_time = None
        decoder_time = None
        if "encoder" in scores and isinstance(scores["encoder"], list):
            encoder_time = scores["encoder"][1]
        if "decoder" in scores and isinstance(scores["decoder"], list):
            decoder_time = scores["decoder"][1]

        summary_rows.append(
            {
                "experiment_name": child.name,
                "experiment_group": group,
                "setting": setting,
                "output_dir": str(child),
                "metrics_dir": str(metrics_dir),
                "psnr": scores.get("psnr"),
                "ssim": scores.get("ssim"),
                "lpips": scores.get("lpips"),
                "encoder_time": encoder_time,
                "decoder_time": decoder_time,
            }
        )

        if scene_path.exists():
            for row in load_json(scene_path):
                per_scene_rows.append(
                    {
                        "experiment_name": child.name,
                        "experiment_group": group,
                        "setting": setting,
                        "scene": row["scene"],
                        "psnr": row["psnr"],
                        "ssim": row["ssim"],
                        "lpips": row["lpips"],
                    }
                )

    summary_json = root / "formal_ablation_summary.json"
    summary_csv = root / "formal_ablation_summary.csv"
    per_scene_json = root / "formal_ablation_per_scene.json"
    per_scene_csv = root / "formal_ablation_per_scene.csv"

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    write_csv(
        summary_csv,
        summary_rows,
        [
            "experiment_name",
            "experiment_group",
            "setting",
            "output_dir",
            "metrics_dir",
            "psnr",
            "ssim",
            "lpips",
            "encoder_time",
            "decoder_time",
        ],
    )

    with per_scene_json.open("w", encoding="utf-8") as handle:
        json.dump(per_scene_rows, handle, indent=2)
    write_csv(
        per_scene_csv,
        per_scene_rows,
        [
            "experiment_name",
            "experiment_group",
            "setting",
            "scene",
            "psnr",
            "ssim",
            "lpips",
        ],
    )

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {per_scene_json}")
    print(f"Wrote {per_scene_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
