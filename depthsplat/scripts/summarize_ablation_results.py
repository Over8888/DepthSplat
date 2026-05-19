import csv
import json
import sys
from pathlib import Path


SKIP_DIRS = {"metrics", "videos", "gaussians", "images", "report"}
ALLOWED_DEPTHCANDS = {"64", "96", "128"}


def parse_setting(path: Path) -> tuple[str, str]:
    name = path.name
    if "noise_translation_" in name:
        return "camera_noise_translation", name.split("noise_translation_", 1)[1]
    if "noise_rotation_" in name:
        return "camera_noise_rotation", name.split("noise_rotation_", 1)[1]
    if "depthcands_" in name:
        return "num_depth_candidates", name.split("depthcands_", 1)[1]
    if name.endswith("baseline"):
        return "baseline", "num_depth_candidates=128"
    return "unknown", name


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/summarize_ablation_results.py <outputs_root>")
        return 1

    root = Path(sys.argv[1]).resolve()
    summary_rows = []
    per_scene_rows = []

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue

        metrics_dir = child / "metrics"
        score_path = metrics_dir / "scores_all_avg.json"
        scene_path = metrics_dir / "scores_per_scene.json"
        if not score_path.exists():
            print(f"[WARN] Missing scores file: {score_path}")
            continue

        scores = load_json(score_path)
        group, setting = parse_setting(child)
        if group == "num_depth_candidates" and setting not in ALLOWED_DEPTHCANDS:
            continue
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
        else:
            print(f"[WARN] Missing per-scene scores file: {scene_path}")

    summary_json = root / "ablation_summary.json"
    summary_csv = root / "ablation_summary.csv"
    per_scene_json = root / "ablation_per_scene.json"
    per_scene_csv = root / "ablation_per_scene.csv"

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
