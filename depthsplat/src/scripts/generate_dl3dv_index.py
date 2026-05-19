import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

DEFAULT_DATASET_PATH = Path("/root/RealEstate10K/")


def generate_index(dataset_root: Path, stages: list[str]) -> None:
    for stage_name in stages:
        stage = dataset_root / stage_name

        index = {}
        for chunk_path in tqdm(
            sorted(stage.iterdir()), desc=f"Indexing {stage.name}"
        ):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage))

        with (stage / "index.json").open("w") as f:
            json.dump(index, f)

        print(f"Saved {len(index)} entries to {stage / 'index.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate index.json for a dataset split made of .torch chunks."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Dataset root containing train/test directories.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["test"],
        help='Split names to index, e.g. "test" or "train test".',
    )
    args = parser.parse_args()

    generate_index(args.dataset_root, args.stages)
