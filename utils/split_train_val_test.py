from pathlib import Path
import shutil
import sys
from typing import List

from sklearn.model_selection import train_test_split
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
    create_path_if_not_exists(output_labels_folder_path / split_name)
    create_path_if_not_exists(output_images_folder_path / split_name)
    split_images_paths = []
    for split_image_filename in split_filenames_paths:
        image_filename = split_image_filename.split("/")[-1]
        image_file_path = output_images_folder_path / split_name / image_filename
        shutil.copyfile(src=images_folder_path / image_filename,
                        dst=image_file_path)
        split_images_paths.append(image_file_path)

        label_path = split_image_filename.replace("images", "labels").replace(".jpg", ".txt")
        label_filename = label_path.split("/")[-1]
        label_file_path = output_labels_folder_path / split_name / label_filename
        shutil.copyfile(src=labels_folder_path / label_filename,
                        dst=label_file_path)
    split_images_files_path = output_images_folder_path.parent / f"{split_name}.txt"
    with open(split_images_files_path, "w") as split_images_list_file:
        for image_path in split_images_paths:
            split_images_list_file.write(str(image_path) + "\n")


images_folder = ROOT_DIR / Path("data/frames")
output_dataset_folder = ROOT_DIR / Path("data/zebra_dataset")
labels_data_path = ROOT_DIR / Path("data/zebra_exported_dataset")
labels_folder = labels_data_path / "labels/Train/frames"
data_yaml_filename = labels_data_path / "data.yaml"
train_filenames_filename = labels_data_path / "Train.txt"

create_path_if_not_exists(output_dataset_folder)

with open(train_filenames_filename, "r") as train_filenames_file:
    train_paths = train_filenames_file.read().splitlines()

train_val_paths = []
test_filenames_paths = []
for image_path in train_paths:
    image_filename = image_path.split("/")[-1]
    if image_filename.startswith("4_"):
        test_filenames_paths.append(image_path)
    elif not image_filename.startswith("4_1"):
        train_val_paths.append(image_path)

train_filenames_paths, val_filenames_paths = train_test_split(train_val_paths,
                                                              test_size=0.25,
                                                              shuffle=True)

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

output_yaml = {"train": "Train.txt", "val": "Val.txt", "test": "Test.txt"}
output_yaml["names"] = dataset_yaml["names"]
output_yaml["path"] = "."

with open(output_dataset_folder / "data.yaml", 'w') as file:
    yaml.dump(output_yaml, file)
