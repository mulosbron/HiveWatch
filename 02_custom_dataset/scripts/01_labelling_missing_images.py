"""
This script performs object detection using a pre-trained YOLO model on all image files found in subdirectories.
It processes each image, applies the YOLO model to detect objects, and saves:
1. A labeled `.txt` file in YOLO format containing bounding box and class data.
2. An annotated copy of the image with drawn bounding boxes and class labels.
The results are saved in parallel output folders next to the original directories.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "model_path": os.path.join(CURRENT_DIR, "..", "..", "03_model_training", "runs", "detect", "pollen_varroa_model_50_640_pre",
                               "weights", "best.pt"),
    "base_dir": os.path.join(CURRENT_DIR, "missing_images_copy"),
    "conf_threshold": 0.6,
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
    "output_folder_suffix": "_yolo",
}


def process_images_in_directory(model_path, base_dir, conf_threshold=0.4):
    try:
        print("[INFO] Loading YOLO model")
        try:
            model = YOLO(model_path)
            print("[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Model could not be loaded: {e}")
            return

        total_processed = 0
        total_folders = 0

        for root, dirs, files in os.walk(base_dir):
            image_files = [file for file in files if os.path.splitext(file.lower())[1] in CONFIG["image_extensions"]]

            if not image_files:
                continue

            total_folders += 1
            print(f"[INFO] Found {len(image_files)} images in folder: {root}")

            folder_name = os.path.basename(root)
            yolo_folder = os.path.join(os.path.dirname(root), f"{folder_name}{CONFIG['output_folder_suffix']}")
            images_output_folder = os.path.join(yolo_folder, "images")
            labels_output_folder = os.path.join(yolo_folder, "labels")

            try:
                os.makedirs(images_output_folder, exist_ok=True)
                os.makedirs(labels_output_folder, exist_ok=True)
                print(f"[INFO] Output folders created: {images_output_folder} and {labels_output_folder}")
            except Exception as e:
                print(f"[ERROR] Could not create output directories: {e}")
                continue

            success_count = 0
            error_count = 0

            for i, image_file in enumerate(image_files):
                try:
                    image_path = os.path.join(root, image_file)
                    print(f"[PROG] ({i + 1}/{len(image_files)}) Processing: {image_file}")

                    frame = cv2.imread(image_path)
                    if frame is None:
                        print(f"[ERROR] Image could not be read: {image_path}")
                        error_count += 1
                        continue

                    height, width, _ = frame.shape

                    results = model(frame, conf=conf_threshold, verbose=False)

                    base_name = os.path.splitext(image_file)[0]
                    annotated_output_path = os.path.join(images_output_folder, f"{base_name}.jpg")
                    label_output_path = os.path.join(labels_output_folder, f"{base_name}.txt")

                    class_colors = {}

                    with open(label_output_path, 'w') as label_file:
                        detection_count = 0
                        if results and results[0].boxes:
                            for box in results[0].boxes:
                                conf = float(box.conf[0].item())
                                cls_id = int(box.cls[0].item())
                                cls_name = model.names.get(cls_id, f"ID:{cls_id}")

                                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())

                                x_center = ((x1 + x2) / 2) / width
                                y_center = ((y1 + y2) / 2) / height
                                w = (x2 - x1) / width
                                h = (y2 - y1) / height

                                label_file.write(f"{cls_id} {x_center} {y_center} {w} {h}\n")
                                detection_count += 1

                                if cls_id not in class_colors:
                                    class_colors[cls_id] = (
                                        np.random.randint(0, 256),
                                        np.random.randint(0, 256),
                                        np.random.randint(0, 256),
                                    )
                                color = class_colors[cls_id]

                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                                label = f"{cls_name} {conf:.2f}"

                                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                                                      0.5, 1)

                                label_y = int(y1) - 10 if int(y1) - 10 > text_height else int(y1) + text_height + 10
                                cv2.rectangle(
                                    frame,
                                    (int(x1), label_y - text_height - baseline),
                                    (int(x1) + text_width, label_y + baseline),
                                    color,
                                    -1,
                                )

                                cv2.putText(
                                    frame,
                                    label,
                                    (int(x1), label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )

                    try:
                        success = cv2.imwrite(annotated_output_path, frame)
                        if not success:
                            print(f"[ERROR] Image could not be saved: {annotated_output_path}")
                            error_count += 1
                        else:
                            success_count += 1
                            total_processed += 1
                    except Exception as e:
                        print(f"[ERROR] An exception occurred while writing the image: {annotated_output_path}, {e}")
                        error_count += 1

                except Exception as e:
                    print(f"[ERROR] Failed to process image {image_file}: {e}")
                    error_count += 1

            print(f"[INFO] Folder completed: {success_count} images processed successfully, {error_count} errors")

        print(f"\n[INFO] Total of {total_processed} images processed across {total_folders} folders")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during processing: {e}")


def main():
    try:
        print("[INFO] Starting YOLO image processing")

        if not os.path.exists(CONFIG["model_path"]):
            print(f"[ERROR] Model file not found: {CONFIG['model_path']}. Please check the 'model_path' in CONFIG.")
            return

        if not os.path.exists(CONFIG["base_dir"]):
            print(f"[ERROR] Base directory not found: {CONFIG['base_dir']}. Please check the 'base_dir' in CONFIG.")
            return

        print(f"[INFO] Base directory: {CONFIG['base_dir']}")
        print(f"[INFO] Model in use: {CONFIG['model_path']}")
        print(f"[INFO] Confidence threshold: {CONFIG['conf_threshold']}")

        process_images_in_directory(
            model_path=CONFIG["model_path"],
            base_dir=CONFIG["base_dir"],
            conf_threshold=CONFIG["conf_threshold"],
        )

        print("[INFO] Program completed successfully")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()