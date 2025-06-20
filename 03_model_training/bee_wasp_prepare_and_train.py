import os
import yaml
import shutil
import random
import traceback
from tqdm import tqdm
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "data_dir": os.path.join(CURRENT_DIR, "..", "02_custom_dataset", "bee_vs_wasp_yolo"),
    "image_extensions": ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif', '.ppm'],
    "model_name": "bee_wasp_model_75_640_sil",
    "epochs": 75,
    "image_size": 640,
}

CONFIG["img_dir"] = os.path.join(CONFIG["data_dir"], "images")
CONFIG["label_dir"] = os.path.join(CONFIG["data_dir"], "labels")
CONFIG["output_dir"] = os.path.join(CONFIG["data_dir"], "yolo_dataset")

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def create_dataset_config():
    os.makedirs(os.path.join(CONFIG["output_dir"], "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "test", "labels"), exist_ok=True)

    all_images = [f for f in os.listdir(CONFIG["img_dir"])
                  if os.path.isfile(os.path.join(CONFIG["img_dir"], f)) and
                  os.path.splitext(f)[1].lower() in CONFIG["image_extensions"]]

    num_images = len(all_images)

    random.shuffle(all_images)

    # 70% train, 20% validation, 10% test
    train_size = int(0.7 * num_images)
    val_size = int(0.2 * num_images)

    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]

    print(f"[INFO] Total number of images: {num_images}")
    print(f"[INFO] Training images: {len(train_images)}")
    print(f"[INFO] Validation images: {len(val_images)}")
    print(f"[INFO] Test images: {len(test_images)}")

    def copy_files(image_list, destination):
        for img_file in tqdm(image_list, desc=f"[PROGRESS] Copying files to {destination} directory"):
            try:
                src_img = os.path.join(CONFIG["img_dir"], img_file)
                dst_img = os.path.join(CONFIG["output_dir"], destination, "images", img_file)

                if not os.path.exists(dst_img):
                    shutil.copy(src_img, dst_img)

                label_file = os.path.splitext(img_file)[0] + ".txt"
                src_label = os.path.join(CONFIG["label_dir"], label_file)
                dst_label = os.path.join(CONFIG["output_dir"], destination, "labels", label_file)

                if os.path.exists(src_label) and not os.path.exists(dst_label):
                    shutil.copy(src_label, dst_label)
            except Exception as e:
                print(f"[ERROR] Error: Problem copying file {img_file}: {e}")

    if len(os.listdir(os.path.join(CONFIG["output_dir"], "train", "images"))) < len(train_images):
        copy_files(train_images, "train")
    else:
        print("[INFO] Training files already copied, skipping.")

    if len(os.listdir(os.path.join(CONFIG["output_dir"], "val", "images"))) < len(val_images):
        copy_files(val_images, "val")
    else:
        print("[INFO] Validation files already copied, skipping.")

    if len(os.listdir(os.path.join(CONFIG["output_dir"], "test", "images"))) < len(test_images):
        copy_files(test_images, "test")
    else:
        print("[INFO] Test files already copied, skipping.")

    with open(os.path.join(CONFIG["label_dir"], "classes.txt"), "r") as f:
        classes = [line.strip() for line in f.readlines()]

    dataset_config = {
        'path': CONFIG["output_dir"],
        'train': os.path.join(CONFIG["output_dir"], "train", "images"),
        'val': os.path.join(CONFIG["output_dir"], "val", "images"),
        'test': os.path.join(CONFIG["output_dir"], "test", "images"),
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes)
    }

    config_path = os.path.join(CONFIG["output_dir"], "dataset.yaml")
    with open(config_path, "w") as f:
        yaml.dump(dataset_config, f)

    print(f"[INFO] Dataset configuration created: {config_path}")

    return config_path, classes


def train_model():
    config_path, classes = create_dataset_config()

    model_folder = os.path.join('runs', 'detect')
    model_name = CONFIG["model_name"]
    model_path = os.path.join(model_folder, model_name, 'weights', 'last.pt')
    resume_training = os.path.exists(model_path)

    print("[INFO] Starting model training...")
    if resume_training:
        print(f"[INFO] Using previous model (transfer learning): {model_path}")
        model = YOLO(model_path)
        resume_param = True
    else:
        print("[INFO] Creating new model...")
        model = YOLO('yolo11x.pt')
        resume_param = False

    model.train(
        data=config_path,
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["image_size"],
        batch=2,
        workers=4,
        device='cuda',
        name=model_name,
        exist_ok=True,
        resume=resume_param,

        augment=True,  # Random augmentation active
        mosaic=1.0,  # Mosaic augmentation (resistance to background noise)
        mixup=0.3,  # Mix-up (clarifies class boundaries)
        hsv_h=0.02,  # Hue augmentation (enhances class discrimination)
        hsv_s=0.8,  # Saturation augmentation
        hsv_v=0.5,  # Brightness augmentation
        flipud=0.2,  # Vertical flipping (class generalization)
        fliplr=0.5,  # Horizontal flipping (class generalization)

        scale=0.7,  # Scaling range (box scale variation)
        translate=0.2,  # Translation ratio (position sensitivity)

        lr0=0.001,  # Initial LR (more aggressive learning)
        lrf=0.01,  # Final LR multiplier (slow end of learning)
        cos_lr=True,  # Cosine LR schedule (stable final stages)
        warmup_epochs=5,  # Warmup duration (initial stability)
        warmup_momentum=0.9,  # Momentum (for generalization)
        warmup_bias_lr=0.1,  # Bias layer LR (fast learning at beginning)

        patience=10  # Early stopping patience (prevent overfitting)
    )

    with open(os.path.join(CONFIG["output_dir"], "classes.txt"), "w") as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    print(f"[INFO] Model training completed. Trained model: runs/train/bee_wasp_model/weights/")


def main():
    print("[INFO] Bee vs WaspHive Object Detection - Model Training")
    print(f"[INFO] Current directory: {CURRENT_DIR}")
    print(f"[INFO] Data directory: {CONFIG['data_dir']}")
    print(f"[INFO] Output directory: {CONFIG['output_dir']}")

    try:
        train_model()
        print("[INFO] Model training completed!")
    except Exception as e:
        print(f"[ERROR] Error during program execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()