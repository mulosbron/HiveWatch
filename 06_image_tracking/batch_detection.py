import os
import cv2
from ultralytics import YOLO
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "model_path": os.path.join(CURRENT_DIR, "..", "04_model_training", "runs", "detect", "bee_wasp_model_150_384",
                               "weights", "best.pt"),
    "source_dir": os.path.join(CURRENT_DIR, "source_images"),
    "output_dir": os.path.join(CURRENT_DIR, "results"),
    "conf_threshold": 0.5,
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
}


def detect_image(model_path, image_path, output_path, conf_threshold=0.4):
    model = YOLO(model_path)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    height, width, _ = frame.shape
    results = model(frame, conf=conf_threshold, verbose=False)

    class_colors = {}

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, f"ID:{cls_id}")

            if cls_id not in class_colors:
                class_colors[cls_id] = (
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                    np.random.randint(0, 256)
                )
            color = class_colors[cls_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            cv2.rectangle(
                frame,
                (x1, label_y - text_height - baseline),
                (x1 + text_width, label_y + baseline),
                color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    try:
        success = cv2.imwrite(output_path, frame)
        if not success:
            print(f"[ERROR] Failed to save image: {output_path}")
    except Exception as e:
        print(f"[ERROR] Exception while saving image: {e}")


def process_all_images(model_path, source_folder, output_folder, conf_threshold=0.4):
    if not os.path.isdir(source_folder):
        print(f"[ERROR] Source directory not found: {source_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    image_files = []
    for file in os.listdir(source_folder):
        file_name, file_ext = os.path.splitext(file)
        file_path = os.path.join(source_folder, file)
        if os.path.isfile(file_path) and file_ext.lower() in CONFIG["image_extensions"]:
            image_files.append(file_path)

    if not image_files:
        print(f"[WARNING] No supported image files found in '{source_folder}'!")
        print(f"[INFO] Supported extensions: {', '.join(CONFIG['image_extensions'])}")
        return

    print(f"[INFO] Processing {len(image_files)} images...")

    for i, image_path in enumerate(image_files):
        base_name = os.path.basename(image_path)
        print(f"[PROGRESS] {i + 1}/{len(image_files)} - {base_name}")

        file_name, file_ext = os.path.splitext(base_name)
        output_name = f"output_{file_name}{file_ext}"
        output_path = os.path.join(output_folder, output_name)

        detect_image(model_path, image_path, output_path, conf_threshold)

    print("[INFO] All images processed successfully")


def main():
    print("[INFO] Bee vs Wasp Detection Tool (YOLO)")
    print(f"[INFO] Model: {os.path.basename(os.path.dirname(os.path.dirname(CONFIG['model_path'])))}")

    os.makedirs(CONFIG['source_dir'], exist_ok=True)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    if not os.path.exists(CONFIG["model_path"]):
        print(f"[ERROR] Model not found: {CONFIG['model_path']}")
        return

    process_all_images(
        model_path=CONFIG["model_path"],
        source_folder=CONFIG["source_dir"],
        output_folder=CONFIG["output_dir"],
        conf_threshold=CONFIG["conf_threshold"]
    )


if __name__ == "__main__":
    main()