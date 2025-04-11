import os
import random
import shutil
from pathlib import Path

# Source
BG_IMG_DIR = r""

# Output
IMAGES_DIR = Path("background_images/imgs")
LABELS_DIR = Path("background_images/labels")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def select_random_images(source_dir, count):
    valid_images = [f for f in os.listdir(source_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(valid_images) < count:
        raise ValueError(f"Not enough images in the {source_dir} directory ({len(valid_images)}/{count})")

    return random.sample(valid_images, count)


try:
    bg_images = select_random_images(BG_IMG_DIR, 5)

    print("Files are being copied and txt files are being created...")
    for img_file in bg_images:
        src_dir = BG_IMG_DIR if img_file in bg_images else BG_IMG_DIR
        src_path = os.path.join(src_dir, img_file)

        target_img = IMAGES_DIR / img_file
        target_txt = LABELS_DIR / (os.path.splitext(img_file)[0] + ".txt")

        shutil.copy(src_path, target_img)

        with open(target_txt, 'w') as f:
            pass  # Create empty file

    print(f"A total of {len(bg_images)} background images and txt files were created.")
    print(f"Images saved to: {IMAGES_DIR.resolve()}")
    print(f"Labels saved to: {LABELS_DIR.resolve()}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    input("Process completed. Press Enter to exit...")