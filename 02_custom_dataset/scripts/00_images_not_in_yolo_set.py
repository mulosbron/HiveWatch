"""
This script scans through multiple source directories to identify image files that are missing from a
target directory. For each missing image, it copies the original image into a separate "missing_images_copy" folder
for further inspection or recovery. This is particularly useful for synchronizing datasets or recovering
files that were unintentionally left out.
"""

import os
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "base_dir": os.path.join(CURRENT_DIR, "..", "..", "00_datasets"),
    "target_path": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "images"),
    "source_subdirs": [
        "BeeOrWasp/kaggle_bee_vs_wasp/other_insect",
        "BeeOrWasp/kaggle_bee_vs_wasp/other_noinsect",
    ],
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
    "missing_images_copy_root": os.path.join(CURRENT_DIR, "missing_images_copy"),
}


def main():
    try:
        print("[INFO] Starting missing images detection and copying process")

        source_paths = [os.path.join(CONFIG["base_dir"], subdir) for subdir in CONFIG["source_subdirs"]]
        print(f"[INFO] Configured to check {len(source_paths)} source directories")

        missing_images = []
        missing_image_paths = []

        copy_root = CONFIG["missing_images_copy_root"]
        try:
            os.makedirs(copy_root, exist_ok=True)
            print(f"[INFO] Output directory created/verified: {copy_root}")
        except Exception as e:
            print(f"[ERROR] Could not create target directory: {copy_root}, {e}")
            return

        target_files = set()
        try:
            target_dir = CONFIG["target_path"]
            if not os.path.isdir(target_dir):
                print(f"[WARNING] Target directory not found: {target_dir}. Assuming empty.")
            else:
                target_files = set(os.listdir(target_dir))
                print(f"[INFO] Found {len(target_files)} files in target directory")
        except Exception as e:
            print(f"[ERROR] Could not list target directory: {target_dir}, {e}. Assuming empty.")

        found_images_count = 0
        processed_source_dirs = 0

        for s_path in source_paths:
            folder_name = os.path.basename(s_path)
            try:
                if not os.path.exists(s_path):
                    print(f"[WARNING] Source directory not found: {s_path}. Skipping.")
                    continue
                elif not os.path.isdir(s_path):
                    print(f"[WARNING] '{s_path}' is not a directory. Skipping.")
                    continue

                processed_source_dirs += 1
                files_in_source = os.listdir(s_path)
                images_in_source = 0
                missing_in_current_source = 0

                for file in files_in_source:
                    full_file_path = os.path.join(s_path, file)

                    if not os.path.isfile(full_file_path):
                        continue

                    ext = os.path.splitext(file)[1].lower()

                    if ext in CONFIG["image_extensions"]:
                        images_in_source += 1
                        found_images_count += 1

                        if file not in target_files:
                            new_name_for_tracking = f"{folder_name}_{file}"
                            missing_images.append(new_name_for_tracking)
                            missing_image_paths.append((full_file_path, folder_name, file))
                            missing_in_current_source += 1

                print(
                    f"[INFO] Processed directory: {folder_name} - Found {images_in_source} images, {missing_in_current_source} missing")

            except Exception as e:
                print(f"[ERROR] Could not process directory: {s_path}, {e}")

        print(f"[INFO] Processed {processed_source_dirs}/{len(source_paths)} source directories")
        print(f"[INFO] Total images found in source directories: {found_images_count}")
        print(f"[INFO] Total missing images detected: {len(missing_image_paths)}")

        if not missing_image_paths:
            print("[INFO] No missing images found to copy")
            print("[INFO] Process completed successfully")
            return

        print(f"[INFO] Copying {len(missing_image_paths)} missing images to: {copy_root}")

        copied_count = 0
        error_count = 0

        for i, (src_path, folder_name, original_filename) in enumerate(missing_image_paths):
            dest_folder = os.path.join(copy_root, folder_name)
            try:
                os.makedirs(dest_folder, exist_ok=True)
            except Exception as e:
                print(f"[ERROR] Could not create subdirectory: {dest_folder}, {e}. Skipping file: {src_path}")
                error_count += 1
                continue

            dest_path = os.path.join(dest_folder, original_filename)

            try:
                if os.path.exists(dest_path):
                    print(f"[WARNING] File already exists at destination: {dest_path}. Skipping.")
                    continue

                shutil.copy2(src_path, dest_path)
                copied_count += 1
                print(f"[PROGRESS] ({copied_count}/{len(missing_image_paths)}) Copied: {original_filename}")
            except Exception as e:
                print(f"[ERROR] Could not copy '{src_path}' to '{dest_path}', {e}")
                error_count += 1

        print(f"[INFO] Successfully copied {copied_count} missing images")
        if error_count > 0:
            print(f"[WARNING] {error_count} errors occurred during copying")

        print("[INFO] Process completed")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()