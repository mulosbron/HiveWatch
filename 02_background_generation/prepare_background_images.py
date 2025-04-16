import os
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "bg_img_dir": os.path.join(CURRENT_DIR, "background_images"),
    "output_dir": os.path.join(CURRENT_DIR, "background_images_processed"),
    "images_subdir": "imgs",
    "labels_subdir": "labels",
    "valid_extensions": ['.jpg', '.jpeg', '.png']
}


def get_valid_images(source_dir):
    try:
        if not os.path.exists(source_dir):
            print(f"[ERROR] Source directory does not exist: {source_dir}")
            raise FileNotFoundError(f"Directory not found: {source_dir}")

        valid_images = [f for f in os.listdir(source_dir)
                        if os.path.splitext(f.lower())[1] in CONFIG["valid_extensions"]]

        print(f"[INFO] Found {len(valid_images)} valid images in source directory")

        if not valid_images:
            print(f"[WARNING] No valid images found in the source directory")
            raise ValueError(f"No valid images found in the {source_dir} directory")

        return valid_images

    except Exception as e:
        if not isinstance(e, (ValueError, FileNotFoundError)):
            print(f"[ERROR] Error getting valid images: {e}")
        raise


def main():
    try:
        print("[INFO] Starting background image processing")

        images_dir = os.path.join(CONFIG["output_dir"], CONFIG["images_subdir"])
        labels_dir = os.path.join(CONFIG["output_dir"], CONFIG["labels_subdir"])

        try:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            print(f"[INFO] Output directories created/verified")
        except Exception as e:
            print(f"[ERROR] Failed to create output directories: {e}")
            return

        bg_images = get_valid_images(CONFIG["bg_img_dir"])

        print("[INFO] Copying files and creating label files...")

        copied_count = 0
        total_images = len(bg_images)

        for img_file in bg_images:
            try:
                src_path = os.path.join(CONFIG["bg_img_dir"], img_file)
                target_img = os.path.join(images_dir, img_file)
                target_txt = os.path.join(labels_dir, f"{os.path.splitext(img_file)[0]}.txt")

                shutil.copy(src_path, target_img)

                with open(target_txt, 'w') as f:
                    pass  # Create empty file

                copied_count += 1
                print(f"[PROGRESS] ({copied_count}/{total_images}) Processed: {img_file}")

            except Exception as e:
                print(f"[ERROR] Failed to process {img_file}: {e}")

        print(f"[INFO] A total of {copied_count} background images and txt files were created")
        print(f"[INFO] Images saved to: {os.path.abspath(images_dir)}")
        print(f"[INFO] Labels saved to: {os.path.abspath(labels_dir)}")
        print("[INFO] Process completed successfully")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()