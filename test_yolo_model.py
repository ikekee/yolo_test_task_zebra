"""This module contains a script for validating a YOLO model on test split."""
from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


def main(dataset_path: Path, model_path: Path):
    model = YOLO(model_path)
    model.val(data=dataset_path / "data.yaml",
              split="test")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to dataset for training model."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to model weights (.pt file) for validation."
    )
    args = parser.parse_args()
    main(args.dataset_path, args.model_path)
