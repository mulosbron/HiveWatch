import os
import numpy as np
from ultralytics import YOLO
import cv2
import glob
import time
import traceback

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "model_paths": [
        os.path.join(CURRENT_DIR, "..", "04_model_training", "runs", "detect", "bee_wasp_model_150_384", "weights",
                     "best.pt"),
        os.path.join(CURRENT_DIR, "..", "04_model_training", "runs", "detect", "bee_wasp_model_75_384", "weights",
                     "best.pt"),
        os.path.join(CURRENT_DIR, "..", "04_model_training", "runs", "detect", "bee_wasp_model_50_512", "weights",
                     "best.pt"),
        os.path.join(CURRENT_DIR, "..", "04_model_training", "runs", "detect", "bee_wasp_model_50_448", "weights",
                     "best.pt")
    ],
    "output_dir": os.path.join(CURRENT_DIR, "ensemble_results"),
    "input_dir": os.path.join(CURRENT_DIR, "source_images"),
    "conf_threshold": 0.65,
    "iou_threshold": 0.1,
    "ensemble_method": 'weighted_avg',
    "auto_weights": True,
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
}


class EnsembleDetector:
    def __init__(self, config):
        self.models = []
        self.model_weights = []
        self.conf_thresh = config["conf_threshold"]
        self.iou_thresh = config["iou_threshold"]
        self.ensemble_method = config["ensemble_method"]
        self.model_names = {}
        self.class_names = {}

        for i, model_path in enumerate(config["model_paths"]):
            if not model_path.lower().endswith('.pt'):
                print(f"[WARNING] Skipping non-PT file: {model_path}")
                continue

            try:
                model_dir = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
                model = YOLO(model_path)
                self.models.append(model)
                self.model_names[i] = model_dir

                if not self.class_names:
                    self.class_names = model.names

                self.model_weights.append(1.0)
            except Exception as e:
                print(f"[ERROR] Could not load model: {model_path} - {e}")
                traceback.print_exc()

        if not self.models:
            raise ValueError("[ERROR] No models were successfully loaded!")

        if config["auto_weights"] and self.ensemble_method == 'weighted_avg':
            self._evaluate_and_set_weights()
        else:
            num_models = len(self.models)
            self.model_weights = [1.0 / num_models] * num_models

        sum_weights = sum(self.model_weights)
        if sum_weights > 0:
            self.model_weights = [w / sum_weights for w in self.model_weights]

    def _evaluate_and_set_weights(self):
        model_performances = {
            "bee_wasp_model_150_384": 0.63107,
            "bee_wasp_model_75_384": 0.59029,
            "bee_wasp_model_50_512": 0.60746,
            "bee_wasp_model_50_448": 0.57864,
        }

        weights = []
        for i in range(len(self.models)):
            model_name = self.model_names.get(i, f"Model_{i + 1}")

            weight_assigned = False
            for key in model_performances:
                if key in model_name:
                    weights.append(model_performances[key])
                    weight_assigned = True
                    break

            if not weight_assigned:
                avg_performance = sum(model_performances.values()) / len(model_performances)
                weights.append(avg_performance)

        self.model_weights = weights

    def predict(self, img_path_or_img, save_img=True, output_dir="ensemble_results"):
        is_path = isinstance(img_path_or_img, str)
        if is_path:
            img = cv2.imread(img_path_or_img)
            if img is None:
                print(f"[ERROR] Cannot read image: {img_path_or_img}")
                return [], None
            orig_img = img.copy()
        else:
            img = img_path_or_img
            orig_img = img.copy()

        all_predictions = []

        for i, model in enumerate(self.models):
            model_name = self.model_names.get(i, f"Model_{i + 1}")
            try:
                results = model(img, conf=self.conf_thresh, verbose=False)[0]
                boxes = results.boxes.data.cpu().numpy()

                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box
                        all_predictions.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class': int(cls),
                            'weight': self.model_weights[i],
                            'model_name': model_name
                        })
            except Exception as e:
                print(f"[ERROR] Prediction error with model {model_name}: {e}")

        merged_predictions = self._merge_predictions(all_predictions)
        img_out = orig_img.copy()

        if merged_predictions:
            class_names = self.class_names if self.class_names else {0: "Class_0", 1: "Class_1"}

            for pred in merged_predictions:
                if pred['confidence'] < self.conf_thresh:
                    continue

                x1, y1, x2, y2 = [int(c) for c in pred['bbox']]
                cls = pred['class']
                conf = pred['confidence']

                color = (0, 255, 0)
                class_label_lower = class_names.get(cls, f'CLS_{cls}').lower()
                if class_label_lower == "wasp" or cls == 1:
                    color = (0, 0, 255)

                cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

                label = f"{class_names.get(cls, f'CLS_{cls}')} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size, _ = cv2.getTextSize(label, font, 0.5, 2)
                cv2.rectangle(img_out, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1 - 10), color, -1)
                cv2.putText(img_out, label, (x1, y1 - 10), font, 0.5, (255, 255, 255), 1)

            if save_img and is_path:
                os.makedirs(output_dir, exist_ok=True)
                base_filename = os.path.basename(img_path_or_img)
                output_filename = f"output_{base_filename}"
                output_path = os.path.join(output_dir, output_filename)

                try:
                    cv2.imwrite(output_path, img_out)
                except Exception as e:
                    print(f"[ERROR] Failed to save image: {output_path}")

        return merged_predictions, img_out

    def _merge_predictions(self, predictions):
        if not predictions:
            return []

        groups = self._group_boxes(predictions, iou_thresh=self.iou_thresh)
        merged_boxes = []

        for group in groups:
            if not group: continue

            if self.ensemble_method == 'weighted_avg':
                final_bbox = np.zeros(4)
                final_confidence = 0.0
                class_votes = {}
                total_weight = 0.0

                for det in group:
                    weight = det['weight']
                    total_weight += weight
                    final_bbox += np.array(det['bbox']) * weight
                    final_confidence += det['confidence'] * weight
                    cls = det['class']
                    class_votes[cls] = class_votes.get(cls, 0.0) + weight

                if total_weight > 0:
                    final_bbox = (final_bbox / total_weight).tolist()
                    final_confidence = final_confidence / total_weight
                    final_class = max(class_votes, key=class_votes.get)
                    merged_boxes.append({
                        'bbox': final_bbox,
                        'confidence': final_confidence,
                        'class': final_class
                    })

            elif self.ensemble_method == 'voting':
                class_votes = {}
                bboxes = []
                confidences = []
                for det in group:
                    cls = det['class']
                    class_votes[cls] = class_votes.get(cls, 0) + 1
                    bboxes.append(det['bbox'])
                    confidences.append(det['confidence'])

                if class_votes:
                    final_class = max(class_votes, key=class_votes.get)
                    final_bbox = np.median(np.array(bboxes), axis=0).tolist()
                    final_confidence = max(confidences)
                    merged_boxes.append({
                        'bbox': final_bbox,
                        'confidence': final_confidence,
                        'class': final_class
                    })

            elif self.ensemble_method == 'max':
                best_det = max(group, key=lambda x: x['confidence'])
                merged_boxes.append({
                    'bbox': best_det['bbox'],
                    'confidence': best_det['confidence'],
                    'class': best_det['class']
                })

        return merged_boxes

    def _group_boxes(self, predictions, iou_thresh=0.5):
        if not predictions:
            return []

        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        groups = []
        assigned = [False] * len(predictions)

        for i in range(len(predictions)):
            if assigned[i]:
                continue

            current_group = [predictions[i]]
            assigned[i] = True

            for j in range(i + 1, len(predictions)):
                if assigned[j]:
                    continue

                iou = self._calculate_iou(predictions[i]['bbox'], predictions[j]['bbox'])
                if iou > iou_thresh:
                    current_group.append(predictions[j])
                    assigned[j] = True

            groups.append(current_group)

        return groups

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(xi2 - xi1, 0)
        inter_height = max(yi2 - yi1, 0)
        inter_area = inter_width * inter_height

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    def process_folder(self, input_folder, output_dir):
        if not os.path.isdir(input_folder):
            print(f"[ERROR] Source directory not found: {input_folder}")
            return

        os.makedirs(output_dir, exist_ok=True)

        image_paths = []
        for ext in CONFIG["image_extensions"]:
            pattern = f"*{ext}"
            image_paths.extend(glob.glob(os.path.join(input_folder, pattern)))
            pattern_upper = f"*{ext.upper()}"
            image_paths.extend(glob.glob(os.path.join(input_folder, pattern_upper)))

        image_paths = sorted(list(set(image_paths)))

        if not image_paths:
            print(f"[WARNING] No supported image files found in '{input_folder}'!")
            print(f"[INFO] Supported extensions: {', '.join([ext for ext in CONFIG['image_extensions']])}")
            return

        print(f"[INFO] Processing {len(image_paths)} images...")

        start_time = time.time()
        processed_count = 0
        error_count = 0

        for i, img_path in enumerate(image_paths):
            print(f"[PROGRESS] {i + 1}/{len(image_paths)} - {os.path.basename(img_path)}")
            try:
                _, _ = self.predict(img_path, save_img=True, output_dir=output_dir)
                processed_count += 1
            except Exception as e:
                error_count += 1
                print(f"[ERROR] Failed to process {os.path.basename(img_path)}: {e}")

        print("[INFO] All images processed successfully")


def main():
    print("[INFO] Bee vs Wasp Ensemble Detection Tool (YOLO)")

    models_names = []
    for model_path in CONFIG["model_paths"]:
        if os.path.exists(model_path):
            model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
            models_names.append(model_name)

    if models_names:
        print(f"[INFO] Models: {', '.join(models_names)}")

    valid_models = []
    for model_path in CONFIG["model_paths"]:
        if os.path.exists(model_path):
            valid_models.append(model_path)

    if not valid_models:
        print("[ERROR] No valid model files found. Check paths in CONFIG")
        return

    CONFIG["model_paths"] = valid_models

    if not os.path.exists(CONFIG["input_dir"]):
        print(f"[ERROR] Input directory not found: {CONFIG['input_dir']}")
        return

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    try:
        detector = EnsembleDetector(CONFIG)
        detector.process_folder(
            input_folder=CONFIG["input_dir"],
            output_dir=CONFIG["output_dir"]
        )
    except ValueError as ve:
        print(f"[ERROR] Initialization Error: {ve}")
    except Exception as ex:
        print(f"[ERROR] Unexpected error: {ex}")
        traceback.print_exc()


if __name__ == "__main__":
    main()