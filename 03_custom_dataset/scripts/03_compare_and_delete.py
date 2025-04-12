import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "images_dir": os.path.join(CURRENT_DIR, "img"),
    "labels_dir": os.path.join(CURRENT_DIR, "label"),
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
    "label_extension": ".txt",
}


def get_filenames_without_extension(directory, extensions=None):
    filenames = set()

    if not os.path.exists(directory):
        print(f"WARNING: Directory not found: {directory}")
        return filenames

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)
            if extensions is None or ext.lower() in extensions:
                filenames.add(name)

    return filenames


def delete_mismatched_files(images_dir, labels_dir, image_extensions):
    image_filenames = get_filenames_without_extension(images_dir, image_extensions)
    label_filenames = get_filenames_without_extension(labels_dir, [CONFIG["label_extension"]])
    files_to_delete = label_filenames - image_filenames

    print(f"Total images found: {len(image_filenames)} in directory: {images_dir}")
    print(f"Total labels found: {len(label_filenames)} in directory: {labels_dir}")
    print(f"Total labels to delete: {len(files_to_delete)}")

    deleted_count = 0
    for basename in files_to_delete:
        potential_file = os.path.join(labels_dir, basename + CONFIG["label_extension"])
        if os.path.exists(potential_file):
            try:
                os.remove(potential_file)
                print(f"Deleted: {potential_file}")
                deleted_count += 1
            except Exception as e:
                print(f"ERROR: Could not delete file: {potential_file}, ERROR: {e}")

    return deleted_count


def main():
    print(f"Image directory: {CONFIG['images_dir']}")
    print(f"Label directory: {CONFIG['labels_dir']}")

    if not os.path.exists(CONFIG["images_dir"]):
        print(f"ERROR: Image directory not found: {CONFIG['images_dir']}. Please check the 'images_dir' in CONFIG.")
        return

    if not os.path.exists(CONFIG["labels_dir"]):
        print(f"ERROR: Label directory not found: {CONFIG['labels_dir']}. Please check the 'labels_dir' in CONFIG.")
        return

    count = delete_mismatched_files(
        images_dir=CONFIG["images_dir"],
        labels_dir=CONFIG["labels_dir"],
        image_extensions=CONFIG["image_extensions"],
    )

    print(f"\n--- Process completed. Total {count} label files deleted. ---")


if __name__ == "__main__":
    main()
