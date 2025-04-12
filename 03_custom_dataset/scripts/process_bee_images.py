import os
import json
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def classify_and_copy_images(target_dir, dataset_root, resolutions, classes):
    try:
        os.makedirs(target_dir, exist_ok=True)
        print(f"Target directory: {target_dir}")
    except OSError as e:
        print(f"ERROR: Could not create target directory: {target_dir}, ERROR: {e}")
        return

    json_path = os.path.join(dataset_root, "data.json")
    try:
        if not os.path.isfile(json_path):
            print(f"ERROR: JSON file not found: {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"JSON file loaded: {json_path}")
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON file: {json_path}")
        return
    except Exception as e:
        print(f"ERROR: Could not read JSON file: {json_path}, ERROR: {e}")
        return

    processed_count = 0
    skipped_count = 0

    for res in resolutions:
        src_dir = os.path.join(dataset_root, f"images_{res}")
        print(f"Processing source directory: {src_dir} (resolution: {res})")

        if not os.path.isdir(src_dir):
            print(f"WARNING: Source directory not found: {src_dir}. Skipping resolution.")
            continue

        for filename in os.listdir(src_dir):
            if filename.lower().endswith(".jpeg"):
                original_file_path = os.path.join(src_dir, filename)

                if filename not in labels:
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
                        print(
                            f"WARNING: Error classifying '{filename}' for class '{class_name}': {e}")

                if assigned_class:
                    new_filename = f"{res}_{assigned_class}_{filename}"
                    dest_path = os.path.join(target_dir, new_filename)

                    try:
                        shutil.copy2(original_file_path, dest_path)
                        print(f"Copied: {new_filename}")
                        processed_count += 1
                    except Exception as e:
                        print(f"ERROR: Could not copy '{original_file_path}' to '{dest_path}', ERROR: {e}")
                        skipped_count += 1
                else:
                    skipped_count += 1


def main():
    dataset_root = os.path.join(CURRENT_DIR, "..", "..", "00_datasets", "BeeDataset")
    output_folder_name = "classified_output"
    target_dir = os.path.join(CURRENT_DIR, output_folder_name)

    resolutions = ["300", "150", "200"]
    classes = {
        # "wasp": lambda x: x.get("wasps", False),
        # "bee": lambda x: not any([x.get("cooling", False), x.get("pollen", False), x.get("varroa", False), x.get("wasps", False)]),
        "beewvarroa": lambda x: x.get("varroa", False) and not any(
            [x.get("cooling", False), x.get("pollen", False), x.get("wasps", False)]),
        "beewpollen": lambda x: x.get("pollen", False) and not any(
            [x.get("cooling", False), x.get("varroa", False), x.get("wasps", False)]),
    }

    if not os.path.isdir(dataset_root):
        print(f"ERROR: Dataset root directory not found: {dataset_root}")
    else:
        classify_and_copy_images(target_dir, dataset_root, resolutions, classes)

    print("\n--- Process completed. ---")


if __name__ == "__main__":
    main()
