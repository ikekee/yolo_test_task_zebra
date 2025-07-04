from ultralytics import YOLO


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
results = model.train(data="data/zebra_dataset/data.yaml",
                      epochs=3,
                      val=True,
                      device="mps",
                      name="training_with_augmentation",
                      **augmentation_settings)

print(results)
