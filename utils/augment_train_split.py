import albumentations as A
import cv2


train_images_list_file_path = "../data/zebra_dataset/Train.txt"
with open(train_images_list_file_path, "r") as train_filenames_file:
    train_images_paths = train_filenames_file.read().splitlines()

train_labels = []
for train_image_path in train_images_paths:
    label_path = train_image_path.replace("images", "labels").replace(".jpg", ".txt")
    train_labels.append(label_path)

transforms_list = [
    A.AtLeastOneBBoxRandomCrop(width=256, height=256, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(scale=(0.8, 1.2),      # Zoom in/out by 80-120%
            rotate=(-15, 15),      # Rotate by -15 to +15 degrees
            # translate_percent=(0, 0.1), # Optional: translate by 0-10%
            # shear=(-10, 10),          # Optional: shear by -10 to +10 degrees
            p=0.7
        ),
    A.Perspective(p=0.5)
]
train_transform = A.Compose(transforms_list,
                            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

for image_path, label_path in zip(train_images_paths, train_labels):
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

    augmented = train_transform(image=image, bboxes=bboxes, class_labels=class_labels)
    transformed_image = augmented['image']
    transformed_bboxes = augmented['bboxes']
    transformed_class_labels = augmented['class_labels']

    image_filename = image_path.split("/")[-1]
    new_image_path = image_path.replace(image_filename, f"augmented_{image_filename}")
    new_label_path = new_image_path.replace("images", "labels").replace(".jpg", ".txt")

    cv2.imwrite(new_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
    with open(new_label_path, "w") as new_label_file:
        for label_num, bbox in enumerate(transformed_bboxes):
            new_label_file.write(f"{transformed_class_labels[label_num]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    with open(train_images_list_file_path, "a") as train_filenames_file:
        train_filenames_file.write(new_image_path + "\n")




