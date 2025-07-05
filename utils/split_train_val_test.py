"""This module contains a script for splitting exported dataset into train, val and test sets."""
from argparse import ArgumentParser
from pathlib import Path
import shutil
import sys
from typing import List

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.path_lib import create_path_if_not_exists


def create_split(split_name: str,
                 images_folder_path: Path,
                 labels_folder_path: Path,
                 output_labels_folder_path: Path,
                 output_images_folder_path: Path,
                 split_filenames_paths: List[str]):
    """Creates a split of dataset by provided filenames.

    Args:
        split_name: The name of the split (e.g. "Train", "Val", "Test").
        images_folder_path: A path to folder containing dataset images.
        labels_folder_path: A path to folder containing dataset labels.
        output_labels_folder_path: A path to folder for saving split labels.
        output_images_folder_path: A path to folder for saving split images.
        split_filenames_paths: A list of paths to images for related split.
    """
    create_path_if_not_exists(output_labels_folder_path / split_name)
    create_path_if_not_exists(output_images_folder_path / split_name)
    for split_image_filename in split_filenames_paths:
        image_filename = split_image_filename.split("/")[-1]
        image_file_path = output_images_folder_path / split_name / image_filename
        shutil.copyfile(src=images_folder_path / image_filename,
                        dst=image_file_path)

        label_path = split_image_filename.replace("images", "labels").replace(".jpg", ".txt")
        label_filename = label_path.split("/")[-1]
        label_file_path = output_labels_folder_path / split_name / label_filename
        shutil.copyfile(src=labels_folder_path / label_filename,
                        dst=label_file_path)


def main(path_to_exported_dataset: Path, path_to_frames: Path, path_to_output_dataset: Path):
    images_folder = ROOT_DIR / path_to_frames
    labels_data_path = ROOT_DIR / path_to_exported_dataset
    output_dataset_folder = ROOT_DIR / path_to_output_dataset
    labels_folder = labels_data_path / "labels/Train/frames"
    data_yaml_filename = labels_data_path / "data.yaml"
    train_filenames_filename = labels_data_path / "Train.txt"

    create_path_if_not_exists(output_dataset_folder)

    with open(train_filenames_filename, "r") as train_filenames_file:
        exported_files_paths = train_filenames_file.read().splitlines()

    train_filenames_paths = []
    val_filenames_paths = []
    test_filenames_paths = []
    for image_path in exported_files_paths:
        image_filename = image_path.split("/")[-1]
        if image_filename.startswith("4_"):
            test_filenames_paths.append(image_path)
        elif image_filename.startswith("3_1_"):
            val_filenames_paths.append(image_path)
        else:
            train_filenames_paths.append(image_path)

    output_labels_path = output_dataset_folder / "labels"
    output_images_path = output_dataset_folder / "images"

    create_split("Train",
                 images_folder,
                 labels_folder,
                 output_labels_path,
                 output_images_path,
                 train_filenames_paths)

    create_split("Val",
                 images_folder,
                 labels_folder,
                 output_labels_path,
                 output_images_path,
                 val_filenames_paths)

    create_split("Test",
                 images_folder,
                 labels_folder,
                 output_labels_path,
                 output_images_path,
                 test_filenames_paths)

    loader = yaml.FullLoader
    with open(data_yaml_filename) as dataset_description_file:
        dataset_yaml = yaml.load(dataset_description_file, Loader=loader)

    output_yaml = {
        "train": "images/Train",
        "val": "images/Val",
        "test": "images/Test",
        "names": dataset_yaml["names"],
        "path": str(output_dataset_folder)
    }

    with open(output_dataset_folder / "data.yaml", 'w') as file:
        yaml.dump(output_yaml, file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_exported_dataset",
        type=Path,
        default=Path("data/zebra_exported_dataset"),
        required=False,
        help="Path to labelled dataset exported from CVAT in YOLO format. By default, the "
             "path is set to data/zebra_exported_dataset"
    )
    parser.add_argument(
        "--path_to_frames",
        type=Path,
        default=Path("data/frames"),
        required=False,
        help="Path to folder containing frames from video. By default, "
             "the path is set to data/frames"
    )
    parser.add_argument(
        "--path_to_output_dataset",
        type=Path,
        default=Path("data/zebra_dataset"),
        required=False,
        help="Path to folder for saving split data as a new dataset. By default, "
             "the path is set to data/zebra_dataset"
    )
    args = parser.parse_args()
    main(args.path_to_exported_dataset, args.path_to_frames, args.path_to_output_dataset)
