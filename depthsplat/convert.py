import argparse
import sys
from pathlib import Path
from typing import Iterable, Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm

DEFAULT_INPUT_IMAGE_DIR = Path("/data/scene-rep/Real-Estate-10k")
DEFAULT_INPUT_METADATA_DIR = Path("/data/scene-rep/Real-Estate-10k/metadata/RealEstate10K")
DEFAULT_OUTPUT_DIR = Path("/data/scene-rep/Real-Estate-10k/re10k_pt")

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RealEstate10K extracted frames into .torch chunks.")
    parser.add_argument(
        "--input-image-dir",
        type=Path,
        default=DEFAULT_INPUT_IMAGE_DIR,
        help="Root directory containing train/test scene image folders.",
    )
    parser.add_argument(
        "--input-metadata-dir",
        type=Path,
        default=DEFAULT_INPUT_METADATA_DIR,
        help="Root directory containing train/test scene metadata txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write output .torch chunks to.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("train", "test"),
        default=["train", "test"],
        help="Which stages to process.",
    )
    return parser.parse_args()


def iter_stage_dirs(stage_dir: Path) -> Iterable[Path]:
    if not stage_dir.exists():
        return []
    return sorted(path for path in stage_dir.iterdir() if path.is_dir())


def iter_stage_metadata(stage_dir: Path) -> Iterable[Path]:
    if not stage_dir.exists():
        return []
    return sorted(path for path in stage_dir.iterdir() if path.is_file() and path.suffix == ".txt")


def get_example_keys(
    stage: Literal["test", "train"],
    input_image_dir: Path,
    input_metadata_dir: Path,
) -> list[str]:
    stage_image_dir = input_image_dir / stage
    stage_metadata_dir = input_metadata_dir / stage

    if not stage_image_dir.exists():
        print(f"[WARN] Skip stage '{stage}': image directory not found: {stage_image_dir}")
        return []
    if not stage_metadata_dir.exists():
        print(f"[WARN] Skip stage '{stage}': metadata directory not found: {stage_metadata_dir}")
        return []

    image_keys = set(
        example.name for example in tqdm(iter_stage_dirs(stage_image_dir), desc=f"Indexing images ({stage})")
    )
    metadata_keys = set(
        example.stem
        for example in tqdm(iter_stage_metadata(stage_metadata_dir), desc=f"Indexing metadata ({stage})")
    )

    missing_image_keys = metadata_keys - image_keys
    if len(missing_image_keys) > 0:
        print(
            f"Found metadata but no images for {len(missing_image_keys)} examples in stage '{stage}'.",
            file=sys.stderr,
        )
    missing_metadata_keys = image_keys - metadata_keys
    if len(missing_metadata_keys) > 0:
        print(
            f"Found images but no metadata for {len(missing_metadata_keys)} examples in stage '{stage}'.",
            file=sys.stderr,
        )

    keys = sorted(image_keys & metadata_keys)
    print(f"Found {len(keys)} matched keys in stage '{stage}'.")
    return keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes in a cross-platform way."""
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load image files as raw bytes (do not decode)."""
    return {
        int(path.stem): load_raw(path)
        for path in sorted(example_path.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def load_metadata(example_path: Path) -> Metadata:
    with example_path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    url = lines[0]

    timestamps = []
    cameras = []

    for line in lines[1:]:
        timestamp, *camera = line.split(" ")
        timestamps.append(int(timestamp))
        cameras.append(np.fromstring(",".join(camera), sep=","))

    timestamps_tensor = torch.tensor(timestamps, dtype=torch.int64)
    cameras_tensor = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps_tensor,
        "cameras": cameras_tensor,
    }


def convert_stage(stage: Literal["test", "train"], input_image_dir: Path, input_metadata_dir: Path, output_dir: Path):
    keys = get_example_keys(stage, input_image_dir, input_metadata_dir)
    if not keys:
        return

    chunk_size = 0
    chunk_index = 0
    chunk: list[Example] = []

    def save_chunk():
        nonlocal chunk_size, chunk_index, chunk

        chunk_key = f"{chunk_index:0>6}"
        print(f"Saving chunk {chunk_key} for stage '{stage}' ({chunk_size / 1e6:.2f} MB, {len(chunk)} examples).")
        stage_output_dir = output_dir / stage
        stage_output_dir.mkdir(exist_ok=True, parents=True)
        torch.save(chunk, stage_output_dir / f"{chunk_key}.torch")

        chunk_size = 0
        chunk_index += 1
        chunk = []

    for key in keys:
        image_dir = input_image_dir / stage / key
        metadata_path = input_metadata_dir / stage / f"{key}.txt"
        num_bytes = get_size(image_dir)

        images = load_images(image_dir)
        example = load_metadata(metadata_path)

        missing_timestamps = [timestamp.item() for timestamp in example["timestamps"] if timestamp.item() not in images]
        if missing_timestamps:
            print(
                f"[WARN] Skip key {key} in stage '{stage}': {len(missing_timestamps)} timestamps have no matching image files.",
                file=sys.stderr,
            )
            continue

        example["images"] = [images[timestamp.item()] for timestamp in example["timestamps"]]
        example["key"] = key

        print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
        chunk.append(example)
        chunk_size += num_bytes

        if chunk_size >= TARGET_BYTES_PER_CHUNK:
            save_chunk()

    if chunk_size > 0:
        save_chunk()


def main():
    args = parse_args()
    input_image_dir = args.input_image_dir.resolve()
    input_metadata_dir = args.input_metadata_dir.resolve()
    output_dir = args.output_dir.resolve()

    for stage in args.stages:
        convert_stage(stage, input_image_dir, input_metadata_dir, output_dir)


if __name__ == "__main__":
    main()
