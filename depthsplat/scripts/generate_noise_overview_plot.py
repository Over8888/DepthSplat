import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float:
    if value in (None, ""):
        raise ValueError("Missing numeric value")
    return float(value)


def parse_translation_setting(setting: str) -> float:
    return float(setting.removeprefix("t").replace("p", "."))


def parse_rotation_setting(setting: str) -> float:
    return float(setting.removeprefix("r").replace("p", "."))


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_noise_overview_plot.py <outputs_root>")
        return 1

    root = Path(sys.argv[1]).resolve()
    summary_rows = read_csv(root / "ablation_summary.csv")

    baseline = next(row for row in summary_rows if row["experiment_group"] == "baseline")
    translation_rows = sorted(
        (row for row in summary_rows if row["experiment_group"] == "camera_noise_translation"),
        key=lambda row: parse_translation_setting(row["setting"]),
    )
    rotation_rows = sorted(
        (row for row in summary_rows if row["experiment_group"] == "camera_noise_rotation"),
        key=lambda row: parse_rotation_setting(row["setting"]),
    )

    trans_x = [0.0] + [parse_translation_setting(row["setting"]) for row in translation_rows]
    rot_x = [0.0] + [parse_rotation_setting(row["setting"]) for row in rotation_rows]

    trans_psnr = [to_float(baseline["psnr"])] + [to_float(row["psnr"]) for row in translation_rows]
    trans_ssim = [to_float(baseline["ssim"])] + [to_float(row["ssim"]) for row in translation_rows]
    trans_lpips = [to_float(baseline["lpips"])] + [to_float(row["lpips"]) for row in translation_rows]

    rot_psnr = [to_float(baseline["psnr"])] + [to_float(row["psnr"]) for row in rotation_rows]
    rot_ssim = [to_float(baseline["ssim"])] + [to_float(row["ssim"]) for row in rotation_rows]
    rot_lpips = [to_float(baseline["lpips"])] + [to_float(row["lpips"]) for row in rotation_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("PSNR", trans_psnr, rot_psnr, False),
        ("SSIM", trans_ssim, rot_ssim, False),
        ("LPIPS", trans_lpips, rot_lpips, True),
    ]

    for ax, (metric_name, trans_values, rot_values, lower_is_better) in zip(axes, metrics):
        ax.plot(trans_x, trans_values, marker="o", linewidth=2, label="Translation Noise")
        ax.plot(rot_x, rot_values, marker="s", linewidth=2, label="Rotation Noise")
        ax.set_title(metric_name)
        ax.set_xlabel("Noise Magnitude")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        if lower_is_better:
            ax.text(0.98, 0.04, "Lower is better", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        else:
            ax.text(0.98, 0.04, "Higher is better", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    axes[0].legend(loc="best")
    fig.suptitle("Camera Noise Robustness Overview")
    fig.tight_layout()

    out_dir = root / "report" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "noise_overview_metrics.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
