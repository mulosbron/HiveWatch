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
    "preprocessed_dir": os.path.join(CURRENT_DIR, "..", "02_custom_dataset", "bee_vs_wasp_yolo", "preprocessed_data"),
    "image_extensions": ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif', '.ppm'],
    "model_name": "bee_wasp_model_50_160_pre",
    "epochs": 50,
    "image_size": 160,
    "batch_size": 4,
    "workers": 6,
}

CONFIG["img_dir"] = os.path.join(CONFIG["preprocessed_dir"], "images")
CONFIG["aug_img_dir"] = os.path.join(CONFIG["preprocessed_dir"], "augmented", "images")
CONFIG["label_dir"] = os.path.join(CONFIG["preprocessed_dir"], "labels")
CONFIG["aug_label_dir"] = os.path.join(CONFIG["preprocessed_dir"], "augmented", "labels")
CONFIG["output_dir"] = os.path.join(CONFIG["data_dir"], "yolo_dataset_preprocessed")

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def create_dataset_config():
    os.makedirs(os.path.join(CONFIG["output_dir"], "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "test", "labels"), exist_ok=True)

    # Get all original preprocessed images
    original_images = [f for f in os.listdir(CONFIG["img_dir"])
                       if os.path.isfile(os.path.join(CONFIG["img_dir"], f)) and
                       os.path.splitext(f)[1].lower() in CONFIG["image_extensions"]]

    # Get all augmented images
    augmented_images = []
    if os.path.exists(CONFIG["aug_img_dir"]):
        augmented_images = [f for f in os.listdir(CONFIG["aug_img_dir"])
                            if os.path.isfile(os.path.join(CONFIG["aug_img_dir"], f)) and
                            os.path.splitext(f)[1].lower() in CONFIG["image_extensions"]]

    # Shuffle both sets
    random.shuffle(original_images)
    random.shuffle(augmented_images)

    # Calculate split for original images
    num_orig_images = len(original_images)
    orig_train_size = int(0.7 * num_orig_images)
    orig_val_size = int(0.2 * num_orig_images)

    # Split original images
    orig_train_images = original_images[:orig_train_size]
    orig_val_images = original_images[orig_train_size:orig_train_size + orig_val_size]
    orig_test_images = original_images[orig_train_size + orig_val_size:]

    # Add all augmented images to training
    train_images = orig_train_images + augmented_images
    val_images = orig_val_images
    test_images = orig_test_images

    print(f"[INFO] Total original images: {num_orig_images}")
    print(f"[INFO] Total augmented images: {len(augmented_images)}")
    print(f"[INFO] Training images: {len(train_images)}")
    print(f"[INFO] Validation images: {len(val_images)}")
    print(f"[INFO] Test images: {len(test_images)}")

    def copy_files(image_list, destination, is_augmented=False):
        source_img_dir = CONFIG["aug_img_dir"] if is_augmented else CONFIG["img_dir"]
        source_label_dir = CONFIG["aug_label_dir"] if is_augmented else CONFIG["label_dir"]

        for img_file in tqdm(image_list, desc=f"[PROGRESS] Copying files to {destination} directory"):
            try:
                src_img = os.path.join(source_img_dir, img_file)
                dst_img = os.path.join(CONFIG["output_dir"], destination, "images", img_file)

                if not os.path.exists(dst_img):
                    shutil.copy(src_img, dst_img)

                label_file = os.path.splitext(img_file)[0] + ".txt"
                src_label = os.path.join(source_label_dir, label_file)
                dst_label = os.path.join(CONFIG["output_dir"], destination, "labels", label_file)

                if os.path.exists(src_label) and not os.path.exists(dst_label):
                    shutil.copy(src_label, dst_label)
            except Exception as e:
                print(f"[ERROR] Error: Problem copying file {img_file}: {e}")

    # Copy original training images
    if len(os.listdir(os.path.join(CONFIG["output_dir"], "train", "images"))) < len(train_images):
        # Copy original images to train
        copy_files([img for img in train_images if not img.endswith("_aug_0.jpg") and
                    not img.endswith("_aug_1.jpg") and
                    not img.endswith("_aug_2.jpg")],
                   "train", False)

        # Copy augmented images to train
        copy_files([img for img in train_images if img.endswith("_aug_0.jpg") or
                    img.endswith("_aug_1.jpg") or
                    img.endswith("_aug_2.jpg")],
                   "train", True)
    else:
        print("[INFO] Training files already copied, skipping.")

    # Copy validation images
    if len(os.listdir(os.path.join(CONFIG["output_dir"], "val", "images"))) < len(val_images):
        copy_files(val_images, "val")
    else:
        print("[INFO] Validation files already copied, skipping.")

    # Copy test images
    if len(os.listdir(os.path.join(CONFIG["output_dir"], "test", "images"))) < len(test_images):
        copy_files(test_images, "test")
    else:
        print("[INFO] Test files already copied, skipping.")

    # Read classes.txt from original data directory
    original_classes_path = os.path.join(CONFIG["data_dir"], "labels", "classes.txt")
    with open(original_classes_path, "r") as f:
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

    print("[INFO] Starting model training with preprocessed dataset...")
    if resume_training:
        print(f"[INFO] Using previous model (transfer learning): {model_path}")
        model = YOLO(model_path)
        resume_param = True
    else:
        print("[INFO] Creating new model...")
        model = YOLO('yolo11x.pt')  # Using YOLOv8x for better performance with 1024px images
        resume_param = False

    model.train(
        data=config_path,
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["image_size"],
        batch=CONFIG["batch_size"],
        workers=CONFIG["workers"],
        device='cuda',
        name=model_name,
        exist_ok=True,
        resume=resume_param,

        # Since we already did preprocessing and augmentation, we can reduce some augmentation
        # But we still keep some for additional diversity
        augment=True,
        mosaic=0.5,  # Reduced from 1.0 - some data already augmented
        mixup=0.1,  # Reduced from 0.3
        hsv_h=0.01,  # Reduced from 0.02
        hsv_s=0.5,  # Reduced from 0.8
        hsv_v=0.3,  # Reduced from 0.5
        flipud=0.1,  # Reduced from 0.2
        fliplr=0.3,  # Reduced from 0.5

        scale=0.5,  # Reduced from 0.7
        translate=0.1,  # Reduced from 0.2

        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,  # Reduced from 5 - better data quality needs less warmup
        warmup_momentum=0.9,
        warmup_bias_lr=0.1,

        patience=15  # Increased from 10 - give more time for convergence with better data
    )

    with open(os.path.join(CONFIG["output_dir"], "classes.txt"), "w") as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    print(f"[INFO] Model training completed. Trained model: runs/train/{model_name}/weights/")


def main():
    print("[INFO] Bee vs WaspHive Object Detection - Model Training (Preprocessed Dataset)")
    print(f"[INFO] Current directory: {CURRENT_DIR}")
    print(f"[INFO] Data directory: {CONFIG['data_dir']}")
    print(f"[INFO] Preprocessed data directory: {CONFIG['preprocessed_dir']}")
    print(f"[INFO] Output directory: {CONFIG['output_dir']}")
    print(f"[INFO] Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    print(f"[INFO] Batch size: {CONFIG['batch_size']}")

    try:
        train_model()
        print("[INFO] Model training completed!")
    except Exception as e:
        print(f"[ERROR] Error during program execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()