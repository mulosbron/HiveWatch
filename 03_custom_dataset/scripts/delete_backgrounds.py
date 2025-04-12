import os
import random
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "labels_directory": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "labels"),
    "images_directory": os.path.join(CURRENT_DIR, "..", "bee_vs_wasp_yolo", "images"),
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
}


def optimize_background_files(labels_dir, images_dir):
    background_files_info = []

    try:
        label_files = os.listdir(labels_dir)
    except FileNotFoundError:
        print(f"ERROR: Label directory not found: {labels_dir}")
        return
    except Exception as e:
        print(f"ERROR: Could not access label directory: {labels_dir}, ERROR: {e}")
        return

    for filename in label_files:
        if filename.lower().endswith(".txt"):
            label_path = os.path.join(labels_dir, filename)
            try:
                if os.path.isfile(label_path):
                    if os.path.getsize(label_path) == 0:
                        base_name = os.path.splitext(filename)[0]
                        background_files_info.append({"basename": base_name, "label_path": label_path})
            except Exception as e:
                print(f"WARNING: Could not process file: {label_path}, ERROR: {e}")
                continue

    num_backgrounds_found = len(background_files_info)
    print(f"Total empty (background) label files found: {num_backgrounds_found}")

    if num_backgrounds_found == 0:
        print("No background files found to delete.")
        return

    while True:
        try:
            num_to_delete_str = input(
                f"Enter number of random background images and labels to delete (Maximum: {num_backgrounds_found}): ")
            num_to_delete = int(num_to_delete_str)
            if 0 < num_to_delete <= num_backgrounds_found:
                break
            else:
                print(f"Please enter a number between 1 and {num_backgrounds_found}.")
        except ValueError:
            print("Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\nProcess canceled by user.")
            return

    files_to_delete_info = random.sample(background_files_info, num_to_delete)
    print(f"Selected {num_to_delete} random background files for deletion.")

    deleted_count = 0
    errors = 0
    for file_info in files_to_delete_info:
        base_name = file_info["basename"]
        label_file_to_delete = file_info["label_path"]

        image_path_pattern = os.path.join(images_dir, base_name + '.*')
        potential_image_files = glob.glob(image_path_pattern)

        # Sadece desteklenen uzantılara sahip dosyaları filtrele
        image_files_found = [f for f in potential_image_files if
                             os.path.splitext(f)[1].lower() in CONFIG["image_extensions"]]

        image_file_to_delete = None
        if len(image_files_found) == 1:
            image_file_to_delete = image_files_found[0]
        elif len(image_files_found) > 1:
            print(
                f"WARNING: Multiple images found for '{base_name}': {image_files_found}. Deleting: {image_files_found[0]}")
            image_file_to_delete = image_files_found[0]

        try:
            if os.path.exists(label_file_to_delete):
                os.remove(label_file_to_delete)
                print(f"Deleted label: {label_file_to_delete}")
            else:
                print(f"WARNING: Label file not found: {label_file_to_delete}")

            if image_file_to_delete and os.path.exists(image_file_to_delete):
                os.remove(image_file_to_delete)
                print(f"Deleted image: {image_file_to_delete}")
                deleted_count += 1
            elif image_file_to_delete:
                print(f"WARNING: Image file not found: {image_file_to_delete}")
            else:
                print(
                    f"WARNING: No image found for '{base_name}' in {images_dir}. Only label deleted (if present).")

        except OSError as e:
            print(f"ERROR: Could not delete files for '{base_name}', ERROR: {e}")
            errors += 1
        except Exception as e:
            print(f"ERROR: Unexpected error deleting files for '{base_name}', ERROR: {e}")
            errors += 1

    if errors > 0:
        print(f"Errors occurred during deletion of {errors} files.")
    remaining_backgrounds = num_backgrounds_found - deleted_count
    print(f"Estimated remaining background files: {remaining_backgrounds}")


def main():
    labels_directory = CONFIG["labels_directory"]
    images_directory = CONFIG["images_directory"]

    print(f"Label directory: {labels_directory}")
    print(f"Image directory: {images_directory}")

    if not os.path.exists(labels_directory):
        print(f"ERROR: Label directory not found: {labels_directory}. Please check the 'labels_directory' in CONFIG.")
        return

    if not os.path.exists(images_directory):
        print(f"ERROR: Image directory not found: {images_directory}. Please check the 'images_directory' in CONFIG.")
        return

    optimize_background_files(labels_directory, images_directory)

    print(f"\n--- Process completed. ---")


if __name__ == "__main__":
    main()
