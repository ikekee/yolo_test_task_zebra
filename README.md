# General description
This repository contains a code, dataset annotations and results for solving a task of detecting dishes on a video using YOLOv11.

# Data description
To be added...

# Results description
You can find the video with the best model prediction in the following link:
https://drive.google.com/file/d/1K7cPMzG4QDV5iNS26NZfi9MsKwkPLoa_/view?usp=sharing

To be added...

# How to reproduce
1. Install required libraries using `pip install -r requirements.txt`.
2. Obtain videos and place them to some folder (e.g. `data/videos/`)
3. Use the script `utils/get_frames_from_video.py` for obtaining frames from video:
    ```bash
    python utils/get_frames_from_video.py [-v video_path] [-o output_dir] [-s scaling_factor]
    ```
    ```Flag -v: required```, a path to video for obtaining frames.

    ```Flag -o: required```, a path to folder for saving frames.

    ```Flag -s: optional```, scaling factor for resizing images. By default, the value 0.2 is used.

    For example:
    ```bash
    python utils/get_frames_from_video.py -v data/videos/mov_videos/1.MOV -o data/frames/
    ```
    It's necessary not to process video named `4_1.MOV` as it is not used for making a dataset.
5. Then use the script `utils/split_train_val_test.py` for making a split for the dataset:
    ```bash
    python utils/split_train_val_test.py [--path_to_exported_dataset] [--path_to_frames] [--path_to_output_dataset]
    ```
   
    ```Flag --path_to_exported_dataset: optional```, a path to folder with the exported annotations. By default, a path to the existing folder `data/zebra_exported_dataset` is used. Change this flag with caution.

    ```Flag --path_to_frames: optional```, a path to folder with all frames obtained from the videos. By default, a path `data/frames` is used.

    ```Flag --path_to_output_dataset: optional```, a path to saving resulting dataset. By default, a path `data/zebra_dataset` is used.

    For example:
    ```bash
    python utils/split_train_val_test.py
    ```
7. Add augmented images to the existing dataset by using the script `utils/augment_train_split.py`:
    ```bash
    python utils/augment_train_split.py [--dataset_path] [--output_dataset_path]
    ```

    ```Flag --dataset_path: required```, a path to dataset for augmenting.

    ```Flag --output_dataset_path: optional```, a path to save resulting dataset with the augmented images. If not provided, new images will be added to existing provided dataset.

    For example:
    ```bash
    python utils/augment_train_split.py --dataset_path data/zebra_dataset --output_dataset_path data/zebra_dataset_augmented
    ```
9. Finally, the dataset is ready and you can train the model. This can be done by running the script `train_yolo.py`:
    ```bash
    python train_yolo.py [--dataset_path] [--run_name]
    ```
   
    ```Flag --dataset_path: required```, a path to dataset for training the model.

    ```Flag --run_name: optional```, name for the folder containing training results. By default, Ultralytics framework saves results in the `run` folder.

    For example:
    ```bash
    python train_yolo.py --dataset_path data/zebra_dataset_augmented
    ```
10. For validating trained model on the test dataset script `test_yolo_model.py` should be used:
    ```bash
    python test_yolo_model.py [--dataset_path] [--model_path]
    ```
    
    ```Flag --dataset_path: required```, a path to dataset with the `test` split for validating.

    ```Flag --model_path: required```, a path to model weights to use.

    For example:
    ```bash
    python test_yolo_model.py --dataset_path data/zebra_dataset_augmented --model_path runs/detect/train/weights/best.pt
    ```
11. For creating a video with the prediction results Ultralyctics YOLO CLI should be used. For example:
    ```bash
    yolo predict model=runs/detect/train/weights/best.pt source=data/videos/mov_videos/4.MOV  save=True
    ```