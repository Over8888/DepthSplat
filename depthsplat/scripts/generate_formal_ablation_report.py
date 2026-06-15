import csv
import json
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


def parse_near_far_setting(setting: str) -> tuple[float, float]:
    payload = setting.removeprefix("near_")
    near_text, far_text = payload.split("_far_")
    return float(near_text.replace("p", ".")), float(far_text.replace("p", "."))


def save_line_plot(xs, ys, xlabel: str, ylabel: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_formal_ablation_report.py <outputs_root>")
        return 1

    root = Path(sys.argv[1]).resolve()
    report_dir = root / "report_formal"
    plots_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(root / "formal_ablation_summary.csv")
    baseline = next(row for row in summary_rows if row["experiment_group"] == "baseline")

    near_far_rows = [row for row in summary_rows if row["experiment_group"] == "near_far"]
    near_rows = sorted(
        (row for row in near_far_rows if parse_near_far_setting(row["setting"])[1] == 100.0),
        key=lambda row: parse_near_far_setting(row["setting"])[0],
    )
    far_rows = sorted(
        (row for row in near_far_rows if parse_near_far_setting(row["setting"])[0] == 0.5),
        key=lambda row: parse_near_far_setting(row["setting"])[1],
    )

    near_x = [0.5] + [parse_near_far_setting(row["setting"])[0] for row in near_rows if parse_near_far_setting(row["setting"])[0] != 0.5]
    near_psnr = [to_float(baseline["psnr"])] + [to_float(row["psnr"]) for row in near_rows if parse_near_far_setting(row["setting"])[0] != 0.5]
    near_ssim = [to_float(baseline["ssim"])] + [to_float(row["ssim"]) for row in near_rows if parse_near_far_setting(row["setting"])[0] != 0.5]
    near_lpips = [to_float(baseline["lpips"])] + [to_float(row["lpips"]) for row in near_rows if parse_near_far_setting(row["setting"])[0] != 0.5]

    far_x = [50.0, 100.0, 200.0]
    far_map = {parse_near_far_setting(row["setting"])[1]: row for row in far_rows}
    far_psnr = [to_float(far_map[50.0]["psnr"]), to_float(baseline["psnr"]), to_float(far_map[200.0]["psnr"])]
    far_ssim = [to_float(far_map[50.0]["ssim"]), to_float(baseline["ssim"]), to_float(far_map[200.0]["ssim"])]
    far_lpips = [to_float(far_map[50.0]["lpips"]), to_float(baseline["lpips"]), to_float(far_map[200.0]["lpips"])]

    save_line_plot(near_x, near_psnr, "near", "PSNR", "PSNR vs near (far=100)", plots_dir / "formal_near_psnr.png")
    save_line_plot(near_x, near_ssim, "near", "SSIM", "SSIM vs near (far=100)", plots_dir / "formal_near_ssim.png")
    save_line_plot(near_x, near_lpips, "near", "LPIPS", "LPIPS vs near (far=100)", plots_dir / "formal_near_lpips.png")
    save_line_plot(far_x, far_psnr, "far", "PSNR", "PSNR vs far (near=0.5)", plots_dir / "formal_far_psnr.png")
    save_line_plot(far_x, far_ssim, "far", "SSIM", "SSIM vs far (near=0.5)", plots_dir / "formal_far_ssim.png")
    save_line_plot(far_x, far_lpips, "far", "LPIPS", "LPIPS vs far (near=0.5)", plots_dir / "formal_far_lpips.png")

    report_path = report_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# DepthSplat 2-view RE10K 正式消融实验报告\n\n")
        handle.write("## 实验设置\n\n")
        handle.write("- 数据集: RE10K 2-view evaluation\n")
        handle.write("- 基线 checkpoint: `depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth`\n")
        handle.write("- 基线设置: `near=0.5, far=100`\n")
        handle.write("- 正式消融组: `near/far`\n\n")
        handle.write("## 图表\n\n")
        for plot_name in [
            "formal_near_psnr.png",
            "formal_near_ssim.png",
            "formal_near_lpips.png",
            "formal_far_psnr.png",
            "formal_far_ssim.png",
            "formal_far_lpips.png",
        ]:
            handle.write(f"![{plot_name}](plots/{plot_name})\n\n")
        handle.write("## 表格建议\n\n")
        handle.write("### 固定 far=100，改变 near\n\n")
        handle.write("| near | far | PSNR ↑ | SSIM ↑ | LPIPS ↓ |\n")
        handle.write("|---:|---:|---:|---:|---:|\n")
        handle.write(f"| 0.25 | 100 | {to_float(next(row['psnr'] for row in near_rows if parse_near_far_setting(row['setting'])[0] == 0.25)):.4f} | {to_float(next(row['ssim'] for row in near_rows if parse_near_far_setting(row['setting'])[0] == 0.25)):.4f} | {to_float(next(row['lpips'] for row in near_rows if parse_near_far_setting(row['setting'])[0] == 0.25)):.4f} |\n")
        handle.write(f"| 0.50 | 100 | {to_float(baseline['psnr']):.4f} | {to_float(baseline['ssim']):.4f} | {to_float(baseline['lpips']):.4f} |\n")
        handle.write(f"| 1.00 | 100 | {to_float(next(row['psnr'] for row in near_rows if parse_near_far_setting(row['setting'])[0] == 1.0)):.4f} | {to_float(next(row['ssim'] for row in near_rows if parse_near_far_setting(row['setting'])[0] == 1.0)):.4f} | {to_float(next(row['lpips'] for row in near_rows if parse_near_far_setting(row['setting'])[0] == 1.0)):.4f} |\n\n")
        handle.write("### 固定 near=0.5，改变 far\n\n")
        handle.write("| near | far | PSNR ↑ | SSIM ↑ | LPIPS ↓ |\n")
        handle.write("|---:|---:|---:|---:|---:|\n")
        handle.write(f"| 0.50 | 50  | {to_float(far_map[50.0]['psnr']):.4f} | {to_float(far_map[50.0]['ssim']):.4f} | {to_float(far_map[50.0]['lpips']):.4f} |\n")
        handle.write(f"| 0.50 | 100 | {to_float(baseline['psnr']):.4f} | {to_float(baseline['ssim']):.4f} | {to_float(baseline['lpips']):.4f} |\n")
        handle.write(f"| 0.50 | 200 | {to_float(far_map[200.0]['psnr']):.4f} | {to_float(far_map[200.0]['ssim']):.4f} | {to_float(far_map[200.0]['lpips']):.4f} |\n\n")

    with (report_dir / "report_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "baseline": baseline,
                "near_rows": near_rows,
                "far_rows": far_rows,
            },
            handle,
            indent=2,
        )

    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
