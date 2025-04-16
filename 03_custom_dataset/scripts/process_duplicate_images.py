import os
import shutil
import hashlib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_DIR = os.path.join(PROJECT_ROOT, '03_custom_dataset', 'bee_vs_wasp_yolo', 'images')

CONFIG = {
    "img_dir": IMG_DIR,
    "valid_extensions": ['.jpg', '.jpeg', '.png'],
    "current_dir": CURRENT_DIR
}


def compute_hash(file_path):
    try:
        hash_obj = hashlib.md5()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        print(f"[ERROR] Failed to compute hash for {os.path.basename(file_path)}: {e}")
        return None


def main():
    try:
        print("[INFO] Starting duplicate image detection process")
        print(f"[INFO] Current directory: {CONFIG['current_dir']}")
        print(f"[INFO] Image directory: {CONFIG['img_dir']}")

        if not os.path.exists(CONFIG["img_dir"]):
            print(f"[ERROR] Image directory not found: {CONFIG['img_dir']}")
            return

        try:
            all_files = os.listdir(CONFIG["img_dir"])
            image_files = [f for f in all_files
                           if os.path.splitext(f.lower())[1] in CONFIG["valid_extensions"]]
            print(f"[INFO] Found {len(image_files)} image files out of {len(all_files)} total files")
        except Exception as e:
            print(f"[ERROR] Failed to list files in directory: {e}")
            return

        hash_to_files = {}

        print("[INFO] Processing images and computing hashes...")
        processed_count = 0
        error_count = 0

        for i, img_file in enumerate(image_files, 1):
            file_path = os.path.join(CONFIG["img_dir"], img_file)
            try:
                hash_value = compute_hash(file_path)
                if hash_value:
                    if hash_value not in hash_to_files:
                        hash_to_files[hash_value] = []
                    hash_to_files[hash_value].append(file_path)
                    processed_count += 1

                    if i % 50 == 0 or i == 1 or i == len(image_files):
                        print(f"[PROGRESS] Processed {i}/{len(image_files)} images")
                else:
                    error_count += 1
            except Exception as e:
                print(f"[ERROR] Problem processing file {img_file}: {e}")
                error_count += 1

        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
        duplicate_groups = len(duplicates)
        total_duplicates = sum(len(files) for files in duplicates.values())

        print(
            f"[INFO] Hash computation complete. Found {duplicate_groups} groups of duplicate images ({total_duplicates} files)")

        print("[INFO] Moving duplicate images to hash-named folders...")
        duplicate_count = 0
        moved_count = 0
        move_errors = 0

        for hash_value, files in hash_to_files.items():
            if len(files) > 1:
                duplicate_dir = os.path.join(os.path.dirname(CONFIG["img_dir"]), hash_value)
                try:
                    os.makedirs(duplicate_dir, exist_ok=True)
                except Exception as e:
                    print(f"[ERROR] Could not create directory {hash_value}: {e}")
                    continue

                for file_path in files:
                    try:
                        filename = os.path.basename(file_path)
                        destination = os.path.join(duplicate_dir, filename)
                        shutil.move(file_path, destination)
                        print(f"[INFO] Moved {filename} to {os.path.basename(duplicate_dir)} folder")
                        moved_count += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to move {os.path.basename(file_path)}: {e}")
                        move_errors += 1

                duplicate_count += 1
                if duplicate_count % 5 == 0 or duplicate_count == duplicate_groups:
                    print(f"[PROGRESS] Processed {duplicate_count}/{duplicate_groups} duplicate groups")

        print(f"[INFO] Summary:")
        print(f"[INFO] - Total images processed: {processed_count}")
        print(f"[INFO] - Duplicate groups found: {duplicate_groups}")
        print(f"[INFO] - Total duplicate files: {total_duplicates}")
        print(f"[INFO] - Successfully moved: {moved_count} files")
        print(f"[INFO] - Errors during move: {move_errors}")
        print(f"[INFO] - Duplicate images saved to hash-named folders in: {os.path.dirname(CONFIG['img_dir'])}")
        print(f"[INFO] Duplicate image detection completed successfully")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] A critical error occurred: {str(e)}")