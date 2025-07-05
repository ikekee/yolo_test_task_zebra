"""This module contains a script for training YOLO model with the predefined parameters."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def main(dataset_path: Path, run_name: Optional[str]):
    model = YOLO("yolo11n.pt")
    augmentation_settings = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "bgr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0
    }
    model.train(data=dataset_path / "data.yaml",
                epochs=5,
                val=True,
                device="mps",
                name=run_name,
                **augmentation_settings)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to dataset for training model."
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Name of the run for training the model. This name is used for naming folder with the"
             " training results."
    )
    args = parser.parse_args()
    main(args.dataset_path, args.run_name)
