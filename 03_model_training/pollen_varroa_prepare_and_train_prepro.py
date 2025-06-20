import os
import yaml
import shutil
import random
import traceback
from tqdm import tqdm
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "data_dir": os.path.join(CURRENT_DIR, "..", "02_custom_dataset", "pollen_vs_varroa_yolo"),
    "preprocessed_dir": os.path.join(CURRENT_DIR, "..", "02_custom_dataset", "pollen_vs_varroa_yolo", "preprocessed_data"),
    "image_extensions": ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif', '.ppm'],
    "model_name": "pollen_varroa_model_50_320_pre_new",
    "epochs": 50,
    "image_size": 320,
    "batch_size": 2,
    "workers": 6,
}

CONFIG["img_dir"] = os.path.join(CONFIG["preprocessed_dir"], "images")
CONFIG["aug_img_dir"] = os.path.join(CONFIG["preprocessed_dir"], "augmented", "images")
CONFIG["label_dir"] = os.path.join(CONFIG["preprocessed_dir"], "labels")
CONFIG["aug_label_dir"] = os.path.join(CONFIG["preprocessed_dir"], "augmented", "labels")
CONFIG["output_dir"] = os.path.join(CONFIG["data_dir"], "yolo_dataset_preprocessed")

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def create_dataset_config():
    # create train/val/test directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(CONFIG["output_dir"], split, "images"), exist_ok=True)
        os.makedirs(os.path.join(CONFIG["output_dir"], split, "labels"), exist_ok=True)

    # list original and augmented images
    original_images = [f for f in os.listdir(CONFIG["img_dir"])
                       if os.path.isfile(os.path.join(CONFIG["img_dir"], f)) and
                       os.path.splitext(f)[1].lower() in CONFIG["image_extensions"]]
    augmented_images = []
    if os.path.exists(CONFIG["aug_img_dir"]):
        augmented_images = [f for f in os.listdir(CONFIG["aug_img_dir"])
                            if os.path.isfile(os.path.join(CONFIG["aug_img_dir"], f)) and
                            os.path.splitext(f)[1].lower() in CONFIG["image_extensions"]]

    random.shuffle(original_images)
    random.shuffle(augmented_images)

    # split original
    num_orig = len(original_images)
    orig_train = int(0.7 * num_orig)
    orig_val = int(0.2 * num_orig)

    orig_train_images = original_images[:orig_train]
    orig_val_images = original_images[orig_train:orig_train + orig_val]
    orig_test_images = original_images[orig_train + orig_val:]

    # include augmented only in training
    train_images = orig_train_images + augmented_images
    val_images = orig_val_images
    test_images = orig_test_images

    print(f"[INFO] Total original images: {num_orig}")
    print(f"[INFO] Total augmented images: {len(augmented_images)}")
    print(f"[INFO] Training images: {len(train_images)}")
    print(f"[INFO] Validation images: {len(val_images)}")
    print(f"[INFO] Test images: {len(test_images)}")

    def copy_files(image_list, split, is_aug=False):
        src_img_dir = CONFIG["aug_img_dir"] if is_aug else CONFIG["img_dir"]
        src_lbl_dir = CONFIG["aug_label_dir"] if is_aug else CONFIG["label_dir"]
        for img in tqdm(image_list, desc=f"[PROGRESS] Copying to {split}"):
            try:
                src_img = os.path.join(src_img_dir, img)
                dst_img = os.path.join(CONFIG["output_dir"], split, "images", img)
                if not os.path.exists(dst_img): shutil.copy(src_img, dst_img)

                lbl = os.path.splitext(img)[0] + ".txt"
                src_lbl = os.path.join(src_lbl_dir, lbl)
                dst_lbl = os.path.join(CONFIG["output_dir"], split, "labels", lbl)
                if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                    shutil.copy(src_lbl, dst_lbl)
            except Exception as e:
                print(f"[ERROR] Problem copying {img}: {e}")

    # copy train
    existing_train = len(os.listdir(os.path.join(CONFIG["output_dir"], "train", "images")))
    if existing_train < len(train_images):
        # original
        orig_train_list = [i for i in train_images if i in orig_train_images]
        copy_files(orig_train_list, "train", is_aug=False)
        # augmented
        aug_train_list = [i for i in train_images if i in augmented_images]
        copy_files(aug_train_list, "train", is_aug=True)
    else:
        print("[INFO] Training files already copied, skipping.")

    # copy val
    if len(os.listdir(os.path.join(CONFIG["output_dir"], "val", "images"))) < len(val_images):
        copy_files(val_images, "val")
    else:
        print("[INFO] Validation files already copied, skipping.")

    # copy test
    if len(os.listdir(os.path.join(CONFIG["output_dir"], "test", "images"))) < len(test_images):
        copy_files(test_images, "test")
    else:
        print("[INFO] Test files already copied, skipping.")

    # read class names from original labels
    original_classes = os.path.join(CONFIG["data_dir"], "labels", "classes.txt")
    with open(original_classes, "r") as f:
        classes = [l.strip() for l in f]

    dataset_yaml = {
        'path': CONFIG["output_dir"],
        'train': os.path.join(CONFIG["output_dir"], "train", "images"),
        'val': os.path.join(CONFIG["output_dir"], "val", "images"),
        'test': os.path.join(CONFIG["output_dir"], "test", "images"),
        'names': {i: n for i, n in enumerate(classes)},
        'nc': len(classes)
    }
    cfg_path = os.path.join(CONFIG["output_dir"], "dataset.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(dataset_yaml, f)

    print(f"[INFO] Dataset config created at: {cfg_path}")
    return cfg_path, classes


def train_model():
    config_path, classes = create_dataset_config()

    runs_dir = os.path.join('runs', 'detect')
    model_name = CONFIG["model_name"]
    model_pt = os.path.join(runs_dir, model_name, 'weights', 'last.pt')
    resume = os.path.exists(model_pt)

    print("[INFO] Starting training on preprocessed pollen_vs_varroa dataset...")
    if resume:
        print(f"[INFO] Resuming from {model_pt}")
        model = YOLO(model_pt)
    else:
        print("[INFO] Creating new model from scratch...")
        model = YOLO('yolo11x.pt')

    model.train(
        data=config_path,
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["image_size"],
        batch=CONFIG["batch_size"],
        workers=CONFIG["workers"],
        device='cuda',
        name=model_name,
        exist_ok=True,
        resume=resume,
        # moderate augmentation since data already preprocessed
        augment=True,
        mosaic=0.5,
        mixup=0.1,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.1,
        fliplr=0.3,
        scale=0.5,
        translate=0.1,
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        warmup_momentum=0.9,
        warmup_bias_lr=0.1,
        patience=15
    )

    # save final classes file
    with open(os.path.join(CONFIG["output_dir"], "classes.txt"), "w") as f:
        for c in classes:
            f.write(f"{c}\n")

    print(f"[INFO] Training complete. Model saved under runs/train/{model_name}/weights/")


def main():
    print("[INFO] Pollen vs Varroa Detection - Training (Preprocessed)")
    print(f"[INFO] Current dir: {CURRENT_DIR}")
    print(f"[INFO] Data dir: {CONFIG['data_dir']}")
    print(f"[INFO] Preprocessed dir: {CONFIG['preprocessed_dir']}")
    print(f"[INFO] Output dir: {CONFIG['output_dir']}")
    print(f"[INFO] Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    print(f"[INFO] Batch size: {CONFIG['batch_size']}")

    try:
        train_model()
        print("[INFO] Training finished successfully!")
    except Exception as e:
        print(f"[ERROR] Execution error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
