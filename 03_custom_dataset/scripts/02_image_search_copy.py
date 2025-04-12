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
    image_names = set()

    if not os.path.exists(directory):
        print(f"WARNING: Directory not found: {directory}")
        return image_names

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            if ext.lower() in CONFIG["image_extensions"]:
                image_names.add(filename)

    return image_names


def find_and_copy_images(source_dir, search_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    source_images = get_image_names(source_dir)
    print(f"Found {len(source_images)} images in source directory: {source_dir}")

    if not os.path.exists(search_dir):
        print(f"ERROR: Search directory not found: {search_dir}. Please check the 'search_dir' in CONFIG.")
        return 0

    copied_count = 0

    for image_name in source_images:
        search_path = os.path.join(search_dir, image_name)

        if os.path.exists(search_path):
            target_path = os.path.join(target_dir, image_name)

            try:
                shutil.copy2(search_path, target_path)
                copied_count += 1
            except Exception as e:
                print(f"ERROR: Could not copy image: {image_name}, ERROR: {e}")

    return copied_count


def main():
    print(f"Source directory: {CONFIG['source_dir']}")
    print(f"Search directory: {CONFIG['search_dir']}")
    print(f"Target directory: {CONFIG['target_dir']}")

    if not os.path.exists(CONFIG["source_dir"]):
        print(f"ERROR: Source directory not found: {CONFIG['source_dir']}. Please check the 'source_dir' in CONFIG.")
        return

    if not os.path.exists(CONFIG["search_dir"]):
        print(f"ERROR: Search directory not found: {CONFIG['search_dir']}. Please check the 'search_dir' in CONFIG.")
        return

    copied_count = find_and_copy_images(
        source_dir=CONFIG["source_dir"],
        search_dir=CONFIG["search_dir"],
        target_dir=CONFIG["target_dir"],
    )

    print(f"\n--- Process completed. Total {copied_count} images copied to: {CONFIG['target_dir']} ---")


if __name__ == "__main__":
    main()
