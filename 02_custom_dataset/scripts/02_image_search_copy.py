"""
This script reads image filenames from the source directory defined in CONFIG,
searches for them in the search directory, copies any found files into the
target directory, and logs INFO/WARNING/ERROR messages to the console.
"""

import os
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "source_dir": os.path.join(CURRENT_DIR, "missing_images_copy", "wasp2_yolo", "images"),
    "search_dir": os.path.join(CURRENT_DIR, "missing_images_copy", "wasp2"),
    "target_dir": os.path.join(CURRENT_DIR, "copied_images"),
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
}


def get_image_names(directory):
    try:
        image_names = set()

        if not os.path.exists(directory):
            print(f"[WARNING] Directory not found: {directory}")
            return image_names

        for filename in os.listdir(directory):
            try:
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in CONFIG["image_extensions"]:
                        image_names.add(filename)
            except Exception as e:
                print(f"[ERROR] Error processing file {filename}: {e}")

        return image_names

    except Exception as e:
        print(f"[ERROR] Error getting image names from {directory}: {e}")
        return set()


def find_and_copy_images(source_dir, search_dir, target_dir):
    try:
        try:
            os.makedirs(target_dir, exist_ok=True)
            print(f"[INFO] Target directory created/verified: {target_dir}")
        except Exception as e:
            print(f"[ERROR] Could not create target directory: {target_dir}, {e}")
            return 0

        source_images = get_image_names(source_dir)
        print(f"[INFO] Found {len(source_images)} images in source directory")

        if not os.path.exists(search_dir):
            print(f"[ERROR] Search directory not found: {search_dir}. Please check the 'search_dir' in CONFIG.")
            return 0

        copied_count = 0
        skipped_count = 0
        error_count = 0
        total_images = len(source_images)

        for idx, image_name in enumerate(source_images, 1):
            search_path = os.path.join(search_dir, image_name)

            if os.path.exists(search_path):
                target_path = os.path.join(target_dir, image_name)

                try:
                    if os.path.exists(target_path):
                        print(f"[WARNING] File already exists at destination: {image_name}. Skipping.")
                        skipped_count += 1
                        continue

                    shutil.copy2(search_path, target_path)
                    copied_count += 1
                    print(f"[PROGRESS] ({idx}/{total_images}) Copied: {image_name}")
                except Exception as e:
                    print(f"[ERROR] Could not copy image: {image_name}, {e}")
                    error_count += 1
            else:
                print(f"[WARNING] Image not found in search directory: {image_name}")
                skipped_count += 1

        print(f"[INFO] Copy summary: {copied_count} copied, {skipped_count} skipped, {error_count} errors")
        return copied_count

    except Exception as e:
        print(f"[ERROR] Error in find_and_copy_images function: {e}")
        return 0


def main():
    try:
        print("[INFO] Starting image copy process")

        print(f"[INFO] Source directory: {CONFIG['source_dir']}")
        print(f"[INFO] Search directory: {CONFIG['search_dir']}")
        print(f"[INFO] Target directory: {CONFIG['target_dir']}")

        if not os.path.exists(CONFIG["source_dir"]):
            print(
                f"[ERROR] Source directory not found: {CONFIG['source_dir']}. Please check the 'source_dir' in CONFIG.")
            return

        if not os.path.exists(CONFIG["search_dir"]):
            print(
                f"[ERROR] Search directory not found: {CONFIG['search_dir']}. Please check the 'search_dir' in CONFIG.")
            return

        copied_count = find_and_copy_images(
            source_dir=CONFIG["source_dir"],
            search_dir=CONFIG["search_dir"],
            target_dir=CONFIG["target_dir"],
        )

        print(f"[INFO] Process completed. Total {copied_count} images copied to: {CONFIG['target_dir']}")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()