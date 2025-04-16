import os
import json
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "dataset_root": os.path.join(CURRENT_DIR, "..", "..", "00_datasets", "BeeDataset"),
    "output_folder_name": "classified_output",
    "resolutions": ["300", "150", "200"],
    "classes": {
        "beewvarroa": lambda x: x.get("varroa", False) and not any(
            [x.get("cooling", False), x.get("pollen", False), x.get("wasps", False)]),
        "beewpollen": lambda x: x.get("pollen", False) and not any(
            [x.get("cooling", False), x.get("varroa", False), x.get("wasps", False)]),
    }
}


def classify_and_copy_images(target_dir, dataset_root, resolutions, classes):
    try:
        print("[INFO] Starting image classification and copying process")

        try:
            os.makedirs(target_dir, exist_ok=True)
            print(f"[INFO] Target directory: {target_dir}")
        except OSError as e:
            print(f"[ERROR] Could not create target directory: {target_dir}, {e}")
            return

        json_path = os.path.join(dataset_root, "data.json")
        try:
            if not os.path.isfile(json_path):
                print(f"[ERROR] JSON file not found: {json_path}")
                return

            with open(json_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            print(f"[INFO] JSON file loaded: {json_path}")
        except json.JSONDecodeError:
            print(f"[ERROR] Invalid JSON file: {json_path}")
            return
        except Exception as e:
            print(f"[ERROR] Could not read JSON file: {json_path}, {e}")
            return

        processed_count = 0
        skipped_count = 0
        error_count = 0

        for class_name in classes:
            print(f"[INFO] Class to process: {class_name}")

        for res in resolutions:
            src_dir = os.path.join(dataset_root, f"images_{res}")
            print(f"[INFO] Processing source directory: {src_dir} (resolution: {res})")

            if not os.path.isdir(src_dir):
                print(f"[WARNING] Source directory not found: {src_dir}. Skipping resolution.")
                continue

            files_in_dir = os.listdir(src_dir)
            jpeg_files = [f for f in files_in_dir if f.lower().endswith(".jpeg")]
            print(f"[INFO] Found {len(jpeg_files)} JPEG files in {src_dir}")

            for i, filename in enumerate(jpeg_files):
                original_file_path = os.path.join(src_dir, filename)

                if filename not in labels:
                    print(f"[WARNING] No label found for file: {filename}")
                    skipped_count += 1
                    continue

                img_labels = labels[filename]
                assigned_class = None
                for class_name, condition in classes.items():
                    try:
                        if condition(img_labels):
                            assigned_class = class_name
                            break
                    except Exception as e:
                        print(f"[WARNING] Error classifying '{filename}' for class '{class_name}': {e}")
                        error_count += 1

                if assigned_class:
                    new_filename = f"{res}_{assigned_class}_{filename}"
                    dest_path = os.path.join(target_dir, new_filename)

                    try:
                        shutil.copy2(original_file_path, dest_path)
                        processed_count += 1
                        if processed_count % 10 == 0 or processed_count == 1:
                            print(f"[PROGRESS] Copied {processed_count} files so far")
                    except Exception as e:
                        print(f"[ERROR] Could not copy '{filename}' to '{dest_path}', {e}")
                        error_count += 1
                        skipped_count += 1
                else:
                    print(f"[INFO] File '{filename}' does not match any class criteria")
                    skipped_count += 1

        print(f"[INFO] Classification summary:")
        print(f"[INFO] - Successfully copied: {processed_count} files")
        print(f"[INFO] - Skipped: {skipped_count} files")
        print(f"[INFO] - Errors: {error_count} files")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during classification: {e}")


def main():
    try:
        print("[INFO] Starting image classification program")

        target_dir = os.path.join(CURRENT_DIR, CONFIG["output_folder_name"])

        print(f"[INFO] Dataset root: {CONFIG['dataset_root']}")
        print(f"[INFO] Target directory: {target_dir}")

        if not os.path.isdir(CONFIG["dataset_root"]):
            print(f"[ERROR] Dataset root directory not found: {CONFIG['dataset_root']}")
        else:
            classify_and_copy_images(
                target_dir,
                CONFIG["dataset_root"],
                CONFIG["resolutions"],
                CONFIG["classes"]
            )

        print("[INFO] Process completed successfully")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()