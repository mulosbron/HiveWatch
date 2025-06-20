"""
This script is designed to clean up mismatched YOLO label files in an image-label dataset.
It compares image files and label files by their base filenames (excluding extensions), and deletes
label files that do not have a corresponding image file. This is especially useful for maintaining
consistency in YOLO training datasets.
"""

import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "images_dir": os.path.join(CURRENT_DIR, "..", "pollen_vs_varroa_yolo", "images"),
    "labels_dir": os.path.join(CURRENT_DIR, "..", "pollen_vs_varroa_yolo", "labels"),
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
    "label_extension": ".txt",
}


def get_filenames_without_extension(directory, extensions=None):
    try:
        filenames = set()

        if not os.path.exists(directory):
            print(f"[WARNING] Directory not found: {directory}")
            return filenames

        for filename in os.listdir(directory):
            try:
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    name, ext = os.path.splitext(filename)
                    if extensions is None or ext.lower() in extensions:
                        filenames.add(name)
            except Exception as e:
                print(f"[ERROR] Error processing file {filename}: {e}")

        return filenames

    except Exception as e:
        print(f"[ERROR] Error getting filenames from {directory}: {e}")
        return set()


def delete_mismatched_files(images_dir, labels_dir, image_extensions):
    try:
        image_filenames = get_filenames_without_extension(images_dir, image_extensions)
        label_filenames = get_filenames_without_extension(labels_dir, [CONFIG["label_extension"]])
        files_to_delete = label_filenames - image_filenames

        print(f"[INFO] Total images found: {len(image_filenames)} in directory: {images_dir}")
        print(f"[INFO] Total labels found: {len(label_filenames)} in directory: {labels_dir}")
        print(f"[INFO] Total labels to delete: {len(files_to_delete)}")

        if not files_to_delete:
            print("[INFO] No mismatched label files found.")
            return 0

        deleted_count = 0
        error_count = 0

        print("[INFO] Starting deletion of mismatched label files")

        for i, basename in enumerate(files_to_delete, 1):
            potential_file = os.path.join(labels_dir, basename + CONFIG["label_extension"])
            if os.path.exists(potential_file):
                try:
                    os.remove(potential_file)
                    deleted_count += 1
                    print(f"[PROGRESS] ({i}/{len(files_to_delete)}) Deleted: {basename + CONFIG['label_extension']}")
                except Exception as e:
                    print(f"[ERROR] Could not delete file: {potential_file}, {e}")
                    error_count += 1

        if error_count > 0:
            print(f"[WARNING] {error_count} files could not be deleted")

        return deleted_count

    except Exception as e:
        print(f"[ERROR] Error in delete_mismatched_files function: {e}")
        return 0


def main():
    try:
        print("[INFO] Starting label cleanup process")

        print(f"[INFO] Image directory: {CONFIG['images_dir']}")
        print(f"[INFO] Label directory: {CONFIG['labels_dir']}")

        if not os.path.exists(CONFIG["images_dir"]):
            print(
                f"[ERROR] Image directory not found: {CONFIG['images_dir']}. Please check the 'images_dir' in CONFIG.")
            return

        if not os.path.exists(CONFIG["labels_dir"]):
            print(
                f"[ERROR] Label directory not found: {CONFIG['labels_dir']}. Please check the 'labels_dir' in CONFIG.")
            return

        count = delete_mismatched_files(
            images_dir=CONFIG["images_dir"],
            labels_dir=CONFIG["labels_dir"],
            image_extensions=CONFIG["image_extensions"],
        )

        print(f"[INFO] Process completed. Total {count} label files deleted.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()