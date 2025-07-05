"""This module contains a code for augmenting the training split of the dataset."""
import argparse
from pathlib import Path
import shutil
from typing import Optional

import albumentations as A
import cv2
import yaml


def main(dataset_path: Path, output_dataset_path: Optional[Path]):
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"The dataset path {dataset_path} does not exist. "
            f"Please make sure you have the dataset in the correct location."
        )
    dataset_folder_to_use = Path(dataset_path)
    if output_dataset_path is not None:
        output_dataset_path = Path(output_dataset_path)
        shutil.copytree(dataset_path, output_dataset_path)
        loader = yaml.FullLoader
        with open(dataset_path / "data.yaml") as dataset_description_file:
            dataset_yaml = yaml.load(dataset_description_file, Loader=loader)

        output_yaml = {
            "train": "images/Train",
            "val": "images/Val",
            "test": "images/Test",
            "names": dataset_yaml["names"],
            "path": dataset_yaml["path"].replace(dataset_path.name, output_dataset_path.name),
        }

        with open(output_dataset_path / "data.yaml", 'w') as file:
            yaml.dump(output_yaml, file)
        dataset_folder_to_use = Path(output_dataset_path)

    train_images_path = dataset_folder_to_use / "images/Train"

    transforms_list = [
        A.AtLeastOneBBoxRandomCrop(width=256, height=256, p=1.0),
        A.Rotate(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=0.7),
        A.Perspective(p=0.5)
    ]
    train_transform = A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
        seed=42
    )

    for train_image_path in train_images_path.glob("*"):
        image_path = str(train_image_path)
        label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_labels = []
        bboxes = []
        with open(label_path, "r") as label_file:
            for line in label_file:
                split_line = line.strip().split()
                class_labels.append(int(line[0]))
                bbox = [float(x) for x in split_line[1:]]
                bboxes.append(bbox)
        for i in range(2):
            augmented = train_transform(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = augmented['image']
            transformed_bboxes = augmented['bboxes']
            transformed_class_labels = augmented['class_labels']

            image_filename = image_path.split("/")[-1]
            new_image_path = image_path.replace(image_filename, f"augmented_{i}_{image_filename}")
            new_label_path = new_image_path.replace("images", "labels").replace(".jpg", ".txt")

            cv2.imwrite(new_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
            with open(new_label_path, "w") as new_label_file:
                for label_num, bbox in enumerate(transformed_bboxes):
                    new_label_file.write(
                        f"{transformed_class_labels[label_num]}"
                        f" {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("data/zebra_dataset"),
        help="Path to the dataset folder. By default, it is set to data/zebra_dataset"
    )
    parser.add_argument(
        "--output_dataset_path",
        default=None,
        help="Path to the output dataset folder. If not provided, augmented images will be added to"
             " the original dataset."
    )

    args = parser.parse_args()
    main(args.dataset_path, args.output_dataset_path)
