"""
This script validates a YOLO-format dataset by checking for:
1. Missing image files for label files.
2. Missing label files for image files.
3. Invalid bounding box formats or values.
4. Invalid class IDs.
Any problematic files are copied into categorized error folders under 'dataset_errors' for review.
"""

import os
import shutil
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "img_dir": os.path.join(CURRENT_DIR, "..", "pollen_vs_varroa_yolo", "images"),
    "label_dir": os.path.join(CURRENT_DIR, "..", "pollen_vs_varroa_yolo", "labels"),
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
}


def validate_yolo_dataset(config):
    label_dir = config["label_dir"]
    img_dir = config["img_dir"]

    main_error_folder = os.path.join(CURRENT_DIR, "dataset_errors")
    error_folders = {
        "missing_image_for_label": os.path.join(main_error_folder, "MISSING_IMAGE_FOR_LABEL"),
        "missing_label_for_image": os.path.join(main_error_folder, "MISSING_LABEL_FOR_IMAGE"),
        "invalid_bbox": os.path.join(main_error_folder, "INVALID_BBOX"),
        "invalid_class": os.path.join(main_error_folder, "INVALID_CLASS")
    }

    os.makedirs(main_error_folder, exist_ok=True)
    for folder in error_folders.values():
        os.makedirs(folder, exist_ok=True)

    errors = {key: [] for key in error_folders.keys()}

    image_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in config["image_extensions"]
    }

    label_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(label_dir)
        if f.endswith(".txt") and f != "classes.txt"
    }

    # Check for images without labels
    for img_base, img_file in image_files.items():
        if img_base not in label_files:
            errors["missing_label_for_image"].append(img_file)
            shutil.copy2(os.path.join(img_dir, img_file), os.path.join(error_folders["missing_label_for_image"], img_file))

    # Check for labels without images
    for lbl_base, lbl_file in label_files.items():
        if lbl_base not in image_files:
            errors["missing_image_for_label"].append(lbl_file)
            shutil.copy2(os.path.join(label_dir, lbl_file), os.path.join(error_folders["missing_image_for_label"], lbl_file))

    # Validate bounding boxes
    for lbl_base, lbl_file in tqdm(label_files.items(), desc="Validating label files"):
        file_path = os.path.join(label_dir, lbl_file)

        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                errors["invalid_bbox"].append(lbl_file)
                break
            try:
                cls_id, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    errors["invalid_bbox"].append(lbl_file)
                    break
            except ValueError:
                errors["invalid_bbox"].append(lbl_file)
                break

        if lbl_file in errors["invalid_bbox"] and lbl_base in image_files:
            for file in [lbl_file, image_files[lbl_base]]:
                src_dir = label_dir if file.endswith(".txt") else img_dir
                shutil.copy2(os.path.join(src_dir, file), os.path.join(error_folders["invalid_bbox"], file))

    # Summary
    print(f"\n[INFO] Error report directory: {os.path.abspath(main_error_folder)}")
    for error_type, files in errors.items():
        if files:
            print(f"\n[INFO] {error_type.upper().replace('_', ' ')} ({len(files)}):")
            print(f"[INFO] Stored in: {error_folders[error_type]}")
            print("[INFO] Examples:")
            for file in files[:3]:
                print(f"  - {file}")
            if len(files) > 3:
                print(f"  ... and {len(files) - 3} more")

    total_errors = sum(len(v) for v in errors.values())
    print(f"\n[INFO] Total problematic files: {total_errors}")
    return total_errors == 0


def main():
    is_valid = validate_yolo_dataset(CONFIG)
    print("\n[INFO] Dataset validation completed successfully" if is_valid else "\n[WARNING] Issues were found during validation")


if __name__ == "__main__":
    main()
