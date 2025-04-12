import os
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "base_dir": os.path.join(CURRENT_DIR, "..", "..", "00_datasets"),
    "target_path": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "images"),
    "source_subdirs": [
        "BeeOrWasp/kaggle_bee_vs_wasp/bee1",
        "BeeOrWasp/kaggle_bee_vs_wasp/bee2",
        "BeeOrWasp/kaggle_bee_vs_wasp/other_insect",
        "BeeOrWasp/kaggle_bee_vs_wasp/other_noinsect",
        "BeeOrWasp/kaggle_bee_vs_wasp/wasp1",
        "BeeOrWasp/kaggle_bee_vs_wasp/wasp2",
    ],
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
    "missing_images_copy_root": os.path.join(CURRENT_DIR, "missing_images_copy"),
}


def main():
    source_paths = [os.path.join(CONFIG["base_dir"], subdir) for subdir in CONFIG["source_subdirs"]]

    missing_images = []
    missing_image_paths = []

    copy_root = CONFIG["missing_images_copy_root"]
    try:
        os.makedirs(copy_root, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create target directory: {copy_root}, ERROR: {e}")
        return

    target_files = set()
    try:
        target_dir = CONFIG["target_path"]
        if not os.path.isdir(target_dir):
            print(f"WARNING: Target directory not found: {target_dir}. Assuming empty.")
        else:
            target_files = set(os.listdir(target_dir))
    except Exception as e:
        print(f"ERROR: Could not list target directory: {target_dir}, ERROR: {e}. Assuming empty.")

    found_images_count = 0
    processed_source_dirs = 0
    missing_in_current_source = 0

    for s_path in source_paths:
        folder_name = os.path.basename(s_path)
        try:
            if not os.path.exists(s_path):
                print(f"WARNING: Source directory not found: {s_path}. Skipping.")
                continue
            elif not os.path.isdir(s_path):
                print(f"WARNING: '{s_path}' is not a directory. Skipping.")
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

        except Exception as e:
            print(f"ERROR: Could not process directory: {s_path}, ERROR: {e}")

    print(f"Processed {processed_source_dirs}/{len(source_paths)} source directories.")
    print(f"Total images found in source directories: {found_images_count}")
    print(f"Total missing images detected: {len(missing_image_paths)}")

    if not missing_image_paths:
        print("\nNo missing images found to copy.")
        return

    copied_count = 0
    error_count = 0

    for src_path, folder_name, original_filename in missing_image_paths:
        dest_folder = os.path.join(copy_root, folder_name)
        try:
            os.makedirs(dest_folder, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Could not create subdirectory: {dest_folder}, ERROR: {e}. Skipping file: {src_path}")
            error_count += 1
            continue

        dest_path = os.path.join(dest_folder, original_filename)

        try:
            if os.path.exists(dest_path):
                continue

            shutil.copy2(src_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"ERROR: Could not copy '{src_path}' to '{dest_path}', ERROR: {e}")
            error_count += 1

    if error_count > 0:
        print(f"Errors occurred during copying: {error_count}")


if __name__ == "__main__":
    main()
