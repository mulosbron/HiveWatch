import os
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "image_directory": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "images"),
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
}


def calculate_max_dimension_average(f_path):
    max_dimensions = []

    if not os.path.exists(f_path):
        print(f"ERROR: Directory not found: {f_path}")
        return None

    for filename in os.listdir(f_path):
        file_path = os.path.join(f_path, filename)

        try:
            _, ext = os.path.splitext(filename)
            if ext.lower() in CONFIG["image_extensions"]:
                with Image.open(file_path) as img:
                    width, height = img.size
                    max_dimension = max(width, height)
                    max_dimensions.append(max_dimension)
        except (IOError, OSError):
            continue

    if not max_dimensions:
        print(f"WARNING: No processable images found in directory: {f_path}")
        return None

    average = sum(max_dimensions) / len(max_dimensions)
    return average


def main():
    f_path = CONFIG["image_directory"]

    print(f"Image directory: {f_path}")

    if not os.path.exists(f_path):
        print(f"ERROR: Image directory not found: {f_path}. Please check the 'image_directory' in CONFIG.")
        return

    average_max_dimension = calculate_max_dimension_average(f_path)
    if average_max_dimension is not None:
        print(f"Average maximum dimension of images: {average_max_dimension:.2f} pixels")

    print("\n--- Program Finished ---")


if __name__ == "__main__":
    main()
