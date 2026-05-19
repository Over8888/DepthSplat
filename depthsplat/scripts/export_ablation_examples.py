import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT_DIR = Path("/root/depthsplat")
DATASET_ROOT = Path("/root/autodl-tmp/RealEstate10K")
CHECKPOINT = "pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pick_scenes(rows: list[dict]) -> list[str]:
    baseline = {row["scene"]: row for row in rows if row["experiment_group"] == "baseline"}
    candidates = []
    for row in rows:
        scene = row["scene"]
        if scene not in baseline or row["experiment_group"] == "baseline":
            continue
        if row["experiment_group"] == "camera_noise_translation" and row["setting"] != "t0p10":
            continue
        if row["experiment_group"] == "camera_noise_rotation" and row["setting"] != "r10p0":
            continue
        if row["experiment_group"] == "num_depth_candidates" and row["setting"] != "64":
            continue
        base = baseline[scene]
        delta_psnr = float(base["psnr"]) - float(row["psnr"])
        delta_lpips = float(row["lpips"]) - float(base["lpips"])
        candidates.append((delta_psnr + delta_lpips, scene))
    candidates.sort(reverse=True)
    selected = []
    for _, scene in candidates:
        if scene not in selected:
            selected.append(scene)
        if len(selected) == 5:
            break
    return selected


def build_command(scene: str, output_dir: Path, extra_args: list[str]) -> list[str]:
    return [
        "/root/miniconda3/envs/depthsplat/bin/python",
        "-m",
        "src.main",
        "+experiment=re10k",
        "dataset.test_chunk_interval=1",
        f"dataset.roots=[{DATASET_ROOT}]",
        "model.encoder.num_scales=2",
        "model.encoder.upsample_factor=2",
        "model.encoder.lowest_feature_resolution=4",
        "model.encoder.monodepth_vit_type=vitb",
        "model.encoder.tome.enabled=false",
        f"checkpointing.pretrained_model={CHECKPOINT}",
        "mode=test",
        "test.compute_scores=true",
        "test.metric_chunk_size=16",
        "test.save_image=true",
        "test.save_gt_image=true",
        "test.save_input_images=true",
        "dataset/view_sampler=evaluation",
        "dataset.view_sampler.num_context_views=2",
        "dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json",
        f"dataset.overfit_to_scene={scene}",
        f"output_dir={output_dir}",
        *extra_args,
    ]


def label_image(path: Path, label: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 40), color=(255, 255, 255))
    canvas.paste(image, (0, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), label, fill=(0, 0, 0))
    return canvas


def make_collage(scene_dir: Path, output_path: Path) -> None:
    candidates = [
        ("baseline", scene_dir / "baseline" / "images"),
        ("translation", scene_dir / "translation" / "images"),
        ("rotation", scene_dir / "rotation" / "images"),
        ("depthcands64", scene_dir / "depthcands64" / "images"),
    ]
    first_dir = candidates[0][1] / scene_dir.name / "color"
    image_files = sorted(first_dir.glob("*.png"))
    image_files = [path for path in image_files if not path.name.endswith("_gt.png")]
    if not image_files:
        return
    frame_name = image_files[0].name
    gt_name = frame_name.replace(".png", "_gt.png")

    panels = []
    for label, directory in candidates:
        image_path = directory / scene_dir.name / "color" / frame_name
        if image_path.exists():
            panels.append(label_image(image_path, label))
    gt_path = candidates[0][1] / scene_dir.name / "color" / gt_name
    if gt_path.exists():
        panels.append(label_image(gt_path, "ground_truth"))
    input_dir = candidates[0][1] / scene_dir.name / "color"
    input_images = sorted(input_dir.glob("input_*.png"))
    for index, image_path in enumerate(input_images[:2], start=1):
        panels.append(label_image(image_path, f"context_{index}"))

    if not panels:
        return

    total_width = sum(image.width for image in panels)
    max_height = max(image.height for image in panels)
    collage = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    x = 0
    for image in panels:
        collage.paste(image, (x, 0))
        x += image.width
    collage.save(output_path)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/export_ablation_examples.py <outputs_root>")
        return 1

    root = Path(sys.argv[1]).resolve()
    rows = read_csv(root / "ablation_per_scene.csv")
    scenes = pick_scenes(rows)
    if not scenes:
        print("[WARN] No scenes selected for examples.")
        return 0
    report_examples = root / "report" / "examples"
    report_examples.mkdir(parents=True, exist_ok=True)

    settings = {
        "baseline": ["model.encoder.num_depth_candidates=128"],
        "translation": [
            "model.encoder.num_depth_candidates=128",
            "test.camera_noise.enabled=true",
            "test.camera_noise.apply_to=context",
            "test.camera_noise.mode=translation",
            "test.camera_noise.translation_sigma_ratio=0.10",
            "test.camera_noise.rotation_sigma_deg=0.0",
            "test.camera_noise.seed=777",
        ],
        "rotation": [
            "model.encoder.num_depth_candidates=128",
            "test.camera_noise.enabled=true",
            "test.camera_noise.apply_to=context",
            "test.camera_noise.mode=rotation",
            "test.camera_noise.translation_sigma_ratio=0.0",
            "test.camera_noise.rotation_sigma_deg=10.0",
            "test.camera_noise.seed=777",
        ],
        "depthcands64": ["model.encoder.num_depth_candidates=64"],
    }

    for scene in scenes:
        scene_dir = report_examples / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        for name, extra_args in settings.items():
            output_dir = scene_dir / name
            command = build_command(scene, output_dir, extra_args)
            subprocess.run(command, cwd=ROOT_DIR, check=True)
        make_collage(scene_dir, report_examples / f"{scene}.png")

    with (report_examples / "selected_scenes.txt").open("w", encoding="utf-8") as handle:
        for scene in scenes:
            handle.write(scene + "\n")

    print(f"Wrote examples to {report_examples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
