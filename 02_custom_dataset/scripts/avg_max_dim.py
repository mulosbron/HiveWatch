"""
This script analyzes image files in a specified directory to calculate the average of the maximum dimension
(width or height) of each image. It helps assess the general resolution range of the dataset,
which can be useful for resizing, preprocessing, or model input decisions.
"""

import os
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "image_directory": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "images"),
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
}


def calculate_max_dimension_average(f_path):
    try:
        max_dimensions = []

        if not os.path.exists(f_path):
            print(f"[ERROR] Directory not found: {f_path}")
            return None

        total_files = 0
        processed_files = 0
        skipped_files = 0

        print("[INFO] Analyzing image dimensions")

        files = os.listdir(f_path)

        for filename in files:
            total_files += 1
            file_path = os.path.join(f_path, filename)

            try:
                _, ext = os.path.splitext(filename)
                if ext.lower() in CONFIG["image_extensions"]:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        max_dimension = max(width, height)
                        max_dimensions.append(max_dimension)
                        processed_files += 1

                        if processed_files % 100 == 0:
                            print(f"[PROGRESS] Processed {processed_files} images so far")
                else:
                    skipped_files += 1
            except (IOError, OSError) as e:
                print(f"[WARNING] Could not process image {filename}: {e}")
                skipped_files += 1
                continue
            except Exception as e:
                print(f"[ERROR] Unexpected error processing {filename}: {e}")
                skipped_files += 1
                continue

        if not max_dimensions:
            print(f"[WARNING] No processable images found in directory: {f_path}")
            return None

        print(f"[INFO] Successfully analyzed {processed_files} images")
        print(f"[INFO] Skipped {skipped_files} files")

        average = sum(max_dimensions) / len(max_dimensions)
        return average

    except Exception as e:
        print(f"[ERROR] Error calculating dimension average: {e}")
        return None


def main():
    try:
        print("[INFO] Starting image dimension analysis")

        f_path = CONFIG["image_directory"]

        print(f"[INFO] Image directory: {f_path}")

        if not os.path.exists(f_path):
            print(f"[ERROR] Image directory not found: {f_path}. Please check the 'image_directory' in CONFIG.")
            return

        average_max_dimension = calculate_max_dimension_average(f_path)
        if average_max_dimension is not None:
            print(f"[INFO] Average maximum dimension of images: {average_max_dimension:.2f} pixels")
        else:
            print("[WARNING] Could not calculate average dimension")

        print("[INFO] Program completed successfully")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()