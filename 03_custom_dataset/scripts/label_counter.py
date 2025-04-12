import os
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "data_dir": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo"),
    "label_dir": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "labels"),
}


def count_labels(label_dir):
    bee_count = 0
    wasp_count = 0
    background_count = 0

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    print(f"Label directory: {label_dir}")
    print(f"Sample content of first 3 files:")
    for i, label_file in enumerate(label_files[:3]):
        if os.path.exists(label_file):
            try:
                with open(label_file, 'r') as file:
                    content = file.read().strip()
                    print(f"File {i + 1}: {os.path.basename(label_file)}")
                    print(f"Content: '{content}'")
                    print("-" * 30)
            except Exception as e:
                print(f"WARNING: Could not read file: {label_file}, ERROR: {e}")

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
                            except ValueError:
                                print(f"ERROR: Invalid class ID in file: {label_file}, ID: {parts[0]}")
        except Exception as e:
            print(f"ERROR: Could not read file: {label_file}, ERROR: {e}")

    total_files = len(label_files)
    files_with_labels = total_files - background_count

    return {
        "bee_count": bee_count,
        "wasp_count": wasp_count,
        "background_count": background_count,
        "total_files": total_files,
        "files_with_labels": files_with_labels
    }


def main():
    data_dir = CONFIG["data_dir"]
    label_dir = CONFIG["label_dir"]

    print(f"Data directory: {data_dir}")
    print(f"Label directory: {label_dir}")

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}. Please check the 'data_dir' in CONFIG.")
        exit(1)

    if not os.path.exists(label_dir):
        print(f"ERROR: Label directory not found: {label_dir}. Please check the 'label_dir' in CONFIG.")
        exit(1)

    results = count_labels(label_dir)

    print(f"\nResults:")
    print(f"Total files: {results['total_files']}")
    print(f"Files with labels: {results['files_with_labels']}")
    print(f"Bee labels: {results['bee_count']}")
    print(f"Wasp labels: {results['wasp_count']}")
    print(f"Empty (background) files: {results['background_count']}")

    print(f"\n--- Process completed. ---")


if __name__ == "__main__":
    main()
