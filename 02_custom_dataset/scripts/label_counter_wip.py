"""
This script analyzes YOLO-format label files in a specified directory and counts the occurrences of each class:
bee (0), wasp (1), pollen (2), and varroa (3). It also tracks how many label files are empty (background only).
The script provides a summary of total files, labeled files, and per-class counts, which is helpful for
evaluating dataset balance and distribution.
"""

import os
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "data_dir": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "images"),
    "label_dir": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "labels"),
}


def count_labels(label_dir):
    try:
        print("[INFO] Counting labels in files")

        bee_count = 0
        wasp_count = 0
        pollen_count = 0
        varroa_count = 0
        background_count = 0

        try:
            label_files = glob.glob(os.path.join(label_dir, "*.txt"))
            print(f"[INFO] Found {len(label_files)} label files")
        except Exception as e:
            print(f"[ERROR] Failed to list label files: {e}")
            return None

        print(f"[INFO] Sample content of first 3 files:")
        for i, label_file in enumerate(label_files[:3]):
            if os.path.exists(label_file):
                try:
                    with open(label_file, 'r') as file:
                        content = file.read().strip()
                        print(f"[INFO] File {i + 1}: {os.path.basename(label_file)}")
                        print(f"[INFO] Content: '{content}'")
                except Exception as e:
                    print(f"[WARNING] Could not read file: {label_file}, {e}")

        print("[INFO] Processing all label files")
        processed_count = 0
        error_count = 0

        for label_file in label_files:
            try:
                with open(label_file, 'r') as file:
                    content = file.read().strip()

                    if content == "":
                        background_count += 1
                        continue

                    lines = content.split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split()
                            if parts and len(parts) >= 5:
                                try:
                                    class_id = int(parts[0])
                                    if class_id == 0:
                                        bee_count += 1
                                    elif class_id == 1:
                                        wasp_count += 1
                                    elif class_id == 2:
                                        pollen_count += 1
                                    elif class_id == 3:
                                        varroa_count += 1
                                except ValueError:
                                    print(
                                        f"[ERROR] Invalid class ID in file: {os.path.basename(label_file)}, ID: {parts[0]}")
                                    error_count += 1

                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"[PROGRESS] Processed {processed_count}/{len(label_files)} files")

            except Exception as e:
                print(f"[ERROR] Could not read file: {os.path.basename(label_file)}, {e}")
                error_count += 1

        total_files = len(label_files)
        files_with_labels = total_files - background_count

        print(f"[INFO] Processed {total_files} files with {error_count} errors")

        return {
            "bee_count": bee_count,
            "wasp_count": wasp_count,
            "pollen_count": pollen_count,
            "varroa_count": varroa_count,
            "background_count": background_count,
            "total_files": total_files,
            "files_with_labels": files_with_labels
        }

    except Exception as e:
        print(f"[ERROR] An error occurred while counting labels: {e}")
        return None


def main():
    try:
        print("[INFO] Starting label analysis process")

        data_dir = CONFIG["data_dir"]
        label_dir = CONFIG["label_dir"]

        print(f"[INFO] Data directory: {data_dir}")
        print(f"[INFO] Label directory: {label_dir}")

        if not os.path.exists(data_dir):
            print(f"[ERROR] Data directory not found: {data_dir}. Please check the 'data_dir' in CONFIG.")
            return

        if not os.path.exists(label_dir):
            print(f"[ERROR] Label directory not found: {label_dir}. Please check the 'label_dir' in CONFIG.")
            return

        results = count_labels(label_dir)

        if results:
            print(f"[INFO] Analysis Results:")
            print(f"[INFO] Total files: {results['total_files']}")
            print(f"[INFO] Files with labels: {results['files_with_labels']}")
            print(f"[INFO] Bee labels: {results['bee_count']}")
            print(f"[INFO] Wasp labels: {results['wasp_count']}")
            print(f"[INFO] Pollen labels: {results['pollen_count']}")
            print(f"[INFO] Varroa labels: {results['varroa_count']}")
            print(f"[INFO] Empty (background) files: {results['background_count']}")

        print(f"[INFO] Process completed successfully")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()