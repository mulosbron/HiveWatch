"""
This script renames images and corresponding YOLO label files using full MD5 hashes.
"""

import os
import hashlib
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "images_directory": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "images"),
    "labels_directory": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "labels"),
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
}


def generate_md5(s: str) -> str:
    """Return the full MD5 hash (32 hex chars) of the given string."""
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def rename_dataset(config):
    images_dir = config["images_directory"]
    labels_dir = config["labels_directory"]
    exts = config["image_extensions"]

    # Check that directories exist
    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        return False
    if not os.path.isdir(labels_dir):
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return False

    # Collect all image files
    all_images = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]
    total = len(all_images)
    if total == 0:
        print(f"[WARNING] No images found in: {images_dir}")
        return False

    print(f"[INFO] Found {total} images. Starting renaming process.")

    processed = 0

    for img_name in all_images:
        old_stem, ext = os.path.splitext(img_name)
        old_path = os.path.join(images_dir, img_name)

        try:
            # Generate MD5 hash for the filename
            new_hash = generate_md5(img_name)
            new_name = f"{new_hash}{ext}"
            new_path = os.path.join(images_dir, new_name)

            # Handle rare hash collisions
            if os.path.exists(new_path) and os.path.abspath(old_path) != os.path.abspath(new_path):
                salted = img_name + "_" + new_hash
                new_hash = generate_md5(salted)
                new_name = f"{new_hash}{ext}"
                new_path = os.path.join(images_dir, new_name)

            # Rename image file
            os.rename(old_path, new_path)
            processed += 1
            print(f"[RENAMED] {img_name} → {new_name}")

            # Rename label file if it exists
            old_label = os.path.join(labels_dir, f"{old_stem}.txt")
            new_label = os.path.join(labels_dir, f"{new_hash}.txt")
            if os.path.isfile(old_label):
                os.rename(old_label, new_label)
                print(f"          Label: {old_stem}.txt → {new_hash}.txt")
            else:
                print(f"[WARNING] Label not found: {old_stem}.txt")

        except Exception as e:
            print(f"[ERROR] Could not rename '{img_name}': {e}")
            continue

    print(f"[INFO] Process completed: {processed}/{total} files processed.")
    return True


def main():
    try:
        print("[INFO] Dataset rename utility started.")
        ok = rename_dataset(CONFIG)
        if ok:
            print("[INFO] All files have hashed names now!")
        else:
            print("[WARNING] Some errors occurred. Check logs above.")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
