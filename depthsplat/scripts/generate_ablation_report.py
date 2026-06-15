import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_translation_setting(setting: str) -> float:
    return float(setting.removeprefix("t").replace("p", "."))


def parse_rotation_setting(setting: str) -> float:
    return float(setting.removeprefix("r").replace("p", "."))


def parse_depthcands_setting(setting: str) -> int:
    return int(setting)


def save_line_plot(
    xs,
    ys,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_bar_plot(
    labels,
    values,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_ablation_report.py <outputs_root>")
        return 1

    root = Path(sys.argv[1]).resolve()
    report_dir = root / "report"
    plots_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(root / "ablation_summary.csv")
    baseline_row = next(row for row in summary_rows if row["experiment_group"] == "baseline")
    baseline_psnr = to_float(baseline_row["psnr"])
    baseline_lpips = to_float(baseline_row["lpips"])

    translation_rows = sorted(
        (row for row in summary_rows if row["experiment_group"] == "camera_noise_translation"),
        key=lambda row: parse_translation_setting(row["setting"]),
    )
    rotation_rows = sorted(
        (row for row in summary_rows if row["experiment_group"] == "camera_noise_rotation"),
        key=lambda row: parse_rotation_setting(row["setting"]),
    )
    depthcands_rows = sorted(
        (row for row in summary_rows if row["experiment_group"] == "num_depth_candidates"),
        key=lambda row: parse_depthcands_setting(row["setting"]),
    )

    translation_x = [parse_translation_setting(row["setting"]) for row in translation_rows]
    translation_psnr = [to_float(row["psnr"]) for row in translation_rows]
    translation_lpips = [to_float(row["lpips"]) for row in translation_rows]

    rotation_x = [parse_rotation_setting(row["setting"]) for row in rotation_rows]
    rotation_psnr = [to_float(row["psnr"]) for row in rotation_rows]
    rotation_lpips = [to_float(row["lpips"]) for row in rotation_rows]

    depthcands_x = [parse_depthcands_setting(row["setting"]) for row in depthcands_rows]
    depthcands_psnr = [to_float(row["psnr"]) for row in depthcands_rows]
    depthcands_ssim = [to_float(row["ssim"]) for row in depthcands_rows]
    depthcands_lpips = [to_float(row["lpips"]) for row in depthcands_rows]
    depthcands_encoder_time = [to_float(row["encoder_time"]) for row in depthcands_rows]
    depthcands_decoder_time = [to_float(row["decoder_time"]) for row in depthcands_rows]

    save_line_plot(
        translation_x,
        translation_psnr,
        "Translation noise ratio",
        "PSNR",
        "PSNR vs Translation Noise",
        plots_dir / "translation_noise_psnr.png",
    )
    save_line_plot(
        translation_x,
        translation_lpips,
        "Translation noise ratio",
        "LPIPS",
        "LPIPS vs Translation Noise",
        plots_dir / "translation_noise_lpips.png",
    )
    save_line_plot(
        rotation_x,
        rotation_psnr,
        "Rotation noise (deg)",
        "PSNR",
        "PSNR vs Rotation Noise",
        plots_dir / "rotation_noise_psnr.png",
    )
    save_line_plot(
        rotation_x,
        rotation_lpips,
        "Rotation noise (deg)",
        "LPIPS",
        "LPIPS vs Rotation Noise",
        plots_dir / "rotation_noise_lpips.png",
    )
    save_line_plot(
        depthcands_x,
        depthcands_psnr,
        "num_depth_candidates",
        "PSNR",
        "PSNR vs num_depth_candidates",
        plots_dir / "depthcands_psnr.png",
    )
    save_line_plot(
        depthcands_x,
        depthcands_ssim,
        "num_depth_candidates",
        "SSIM",
        "SSIM vs num_depth_candidates",
        plots_dir / "depthcands_ssim.png",
    )
    save_line_plot(
        depthcands_x,
        depthcands_lpips,
        "num_depth_candidates",
        "LPIPS",
        "LPIPS vs num_depth_candidates",
        plots_dir / "depthcands_lpips.png",
    )
    save_bar_plot(
        [str(x) for x in depthcands_x],
        depthcands_encoder_time,
        "Seconds",
        "Encoder Time vs num_depth_candidates",
        plots_dir / "depthcands_encoder_time.png",
    )
    save_bar_plot(
        [str(x) for x in depthcands_x],
        depthcands_decoder_time,
        "Seconds",
        "Decoder Time vs num_depth_candidates",
        plots_dir / "depthcands_decoder_time.png",
    )

    best_translation_psnr = max(translation_rows, key=lambda row: to_float(row["psnr"]))
    worst_translation_lpips = max(translation_rows, key=lambda row: to_float(row["lpips"]))
    best_depthcands_psnr = max(depthcands_rows, key=lambda row: to_float(row["psnr"]))

    report_path = report_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# DepthSplat 2-view RE10K 消融实验报告\n\n")
        handle.write("## 实验设置\n\n")
        handle.write("- 数据集: RE10K 2-view evaluation\n")
        handle.write("- 基线 checkpoint: `depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth`\n")
        handle.write("- 基线设置: `num_depth_candidates=128`\n")
        handle.write("- 消融组: 相机平移噪声、相机旋转噪声、`num_depth_candidates`\n\n")
        handle.write("## 关键指标\n\n")
        handle.write(f"- Baseline PSNR: {baseline_psnr:.4f}\n")
        handle.write(f"- Baseline LPIPS: {baseline_lpips:.4f}\n")
        handle.write(f"- Translation 组最佳 PSNR: {best_translation_psnr['setting']} -> {float(best_translation_psnr['psnr']):.4f}\n")
        handle.write(f"- Translation 组最差 LPIPS: {worst_translation_lpips['setting']} -> {float(worst_translation_lpips['lpips']):.4f}\n")
        handle.write(f"- num_depth_candidates 组最佳 PSNR: {best_depthcands_psnr['setting']} -> {float(best_depthcands_psnr['psnr']):.4f}\n\n")
        handle.write("## 图表\n\n")
        for plot_name in [
            "translation_noise_psnr.png",
            "translation_noise_lpips.png",
            "rotation_noise_psnr.png",
            "rotation_noise_lpips.png",
            "depthcands_psnr.png",
            "depthcands_ssim.png",
            "depthcands_lpips.png",
            "depthcands_encoder_time.png",
            "depthcands_decoder_time.png",
        ]:
            handle.write(f"![{plot_name}](plots/{plot_name})\n\n")
        examples_dir = report_dir / "examples"
        if examples_dir.exists():
            handle.write("## 代表性样例\n\n")
            for image_path in sorted(examples_dir.glob("*.png")):
                handle.write(f"### {image_path.stem}\n\n")
                handle.write(f"![{image_path.name}](examples/{image_path.name})\n\n")
        handle.write("## 结论摘要\n\n")
        handle.write("- 观察相机噪声曲线时，重点看 PSNR 是否随噪声单调下降，以及 LPIPS 是否明显上升。\n")
        handle.write("- 观察 `num_depth_candidates` 曲线时，重点看候选数减少时性能退化幅度，以及候选数增加时时间开销是否显著上升。\n")
        handle.write("- 若需要和其他方法对比，可直接复用同一汇总格式与作图脚本。\n")

    summary = {
        "baseline": baseline_row,
        "translation_rows": translation_rows,
        "rotation_rows": rotation_rows,
        "depthcands_rows": depthcands_rows,
    }
    with (report_dir / "report_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
