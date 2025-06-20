import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import time
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "model_paths": [
        # os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "pollen_varroa_model_50_320", "weights", "best.pt"),
        os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "pollen_varroa_model_50_320_pre", "weights", "best.pt"),
        os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "pollen_varroa_model_50_448_pre", "weights", "best.pt"),
        os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "pollen_varroa_model_50_640_pre", "weights", "best.pt"),
    ],
    "data_dir": os.path.join(CURRENT_DIR, "..", "02_custom_dataset", "pollen_vs_varroa_yolo"),
    "num_samples": 500,
    "conf_threshold": 0.45,
    "iou_threshold": 0.15,
    "debug": True,
    "debug_sample_size": 5,
    "output_dir": os.path.join(CURRENT_DIR, "pollen_vs_varroa_model_comparison_results"),
}

CONFIG["images_dir"] = os.path.join(CONFIG["data_dir"], "images")
CONFIG["labels_dir"] = os.path.join(CONFIG["data_dir"], "labels")


def load_models():
    models = []
    for model_path in CONFIG["model_paths"]:
        model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        print(f"[INFO] Loading model: {model_name}")
        try:
            model = YOLO(model_path)
            if CONFIG["debug"]:
                print(f"[INFO] Model details - {model_name}")
                print(f"[INFO] - Device: {model.device}")
                print(f"[INFO] - Class names: {model.names}")
                print(f"[INFO] - Task: {model.task}")
            models.append((model_name, model))
        except Exception as e:
            print(f"[ERROR] Failed to load model {model_name}: {e}")
    return models


def get_random_samples(num_samples=500):
    image_paths = glob.glob(os.path.join(CONFIG["images_dir"], "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(CONFIG["images_dir"], "*.jpeg")))
    image_paths.extend(glob.glob(os.path.join(CONFIG["images_dir"], "*.png")))

    if len(image_paths) < num_samples:
        print(f"[WARNING] Only {len(image_paths)} images available, using all of them")
        return image_paths

    sampled_paths = random.sample(image_paths, num_samples)

    if CONFIG["debug"]:
        has_label = 0
        for img_path in sampled_paths:
            label_path = get_label_path(img_path)
            if os.path.exists(label_path):
                has_label += 1

        print(f"[INFO] Out of {len(sampled_paths)} sampled images, {has_label} have matching label files ({has_label / len(sampled_paths) * 100:.1f}%)")

        print("\n[INFO] Checking a few sample images and labels:")
        for i, img_path in enumerate(sampled_paths[:CONFIG["debug_sample_size"]]):
            print(f"\n[INFO] Sample {i + 1}:")
            print(f"[INFO] - Image: {img_path}")

            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Could not read image!")
            else:
                print(f"[INFO] - Image size: {img.shape}")

            label_path = get_label_path(img_path)
            print(f"[INFO] - Label: {label_path}")
            if not os.path.exists(label_path):
                print(f"[WARNING] Label file doesn't exist!")
            else:
                try:
                    boxes = read_yolo_label(label_path)
                    print(f"[INFO] - Label contains {len(boxes)} bounding boxes:")
                    for box in boxes:
                        print(f"[INFO]   Class {int(box[4])}: [{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}]")
                except Exception as e:
                    print(f"[ERROR] Failed to parse label file: {e}")

    return sampled_paths


def get_label_path(image_path):
    img_name = os.path.basename(image_path)
    img_stem = os.path.splitext(img_name)[0]
    label_path = os.path.join(CONFIG["labels_dir"], f"{img_stem}.txt")
    return label_path


def read_yolo_label(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                boxes.append([x1, y1, x2, y2, cls])

    return boxes


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    if union == 0:
        return 0
    return intersection / union


def is_normalized_box(box, threshold=0.9):
    return all(0 <= coord <= threshold for coord in box[:4])


def normalize_box(box, img_width, img_height):
    normalized = box.copy()
    normalized[0] /= img_width
    normalized[1] /= img_height
    normalized[2] /= img_width
    normalized[3] /= img_height
    return normalized


def denormalize_box(box, img_width, img_height):
    denormalized = box.copy()
    denormalized[0] *= img_width
    denormalized[1] *= img_height
    denormalized[2] *= img_width
    denormalized[3] *= img_height
    return denormalized


def match_predictions_with_ground_truth(predictions, ground_truth, iou_threshold=0.5, img_width=None, img_height=None):
    matches = []
    unmatched_gt = list(range(len(ground_truth)))

    if not predictions or not ground_truth:
        return matches, unmatched_gt, []

    debug_matches = []

    processed_preds = []
    processed_gt = []

    gt_is_normalized = all([0 <= coord <= 1 for box in ground_truth for coord in box[:4]])
    pred_is_normalized = all([0 <= coord <= 1 for box in predictions for coord in box[:4]])

    if img_width is not None and img_height is not None:
        for pred in predictions:
            pred_box = pred.copy()
            if pred_is_normalized:
                pred_box[0] *= img_width
                pred_box[1] *= img_height
                pred_box[2] *= img_width
                pred_box[3] *= img_height
            processed_preds.append(pred_box)

        for gt in ground_truth:
            gt_box = gt.copy()
            if gt_is_normalized:
                gt_box[0] *= img_width
                gt_box[1] *= img_height
                gt_box[2] *= img_width
                gt_box[3] *= img_height
            processed_gt.append(gt_box)
    else:
        for pred in predictions:
            pred_box = pred.copy()
            processed_preds.append(pred_box)

        for gt in ground_truth:
            gt_box = gt.copy()
            processed_gt.append(gt_box)

        if gt_is_normalized != pred_is_normalized:
            print("[WARNING] Coordinate system mismatch between ground truth and predictions!")
            print(f"[INFO] Ground truth normalized: {gt_is_normalized}")
            print(f"[INFO] Predictions normalized: {pred_is_normalized}")

    for pred_idx, pred_box in enumerate(processed_preds):
        best_iou = 0
        best_gt_idx = -1
        all_ious = []

        for gt_idx in unmatched_gt:
            gt_box = processed_gt[gt_idx]

            iou = calculate_iou(pred_box[:4], gt_box[:4])
            all_ious.append((gt_idx, iou, gt_box[4]))

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx

        if len(debug_matches) < CONFIG["debug_sample_size"]:
            debug_info = {
                'pred_box': pred_box[:4],
                'pred_class': int(pred_box[4]),
                'pred_conf': pred_box[5] if len(pred_box) > 5 else 0,
                'all_ious': all_ious,
                'best_iou': best_iou,
                'best_gt_idx': best_gt_idx,
                'matched': best_gt_idx >= 0
            }
            debug_matches.append(debug_info)

        if best_gt_idx >= 0:
            matches.append({
                'pred_idx': pred_idx,
                'gt_idx': best_gt_idx,
                'iou': best_iou,
                'pred_class': int(pred_box[4]),
                'gt_class': int(processed_gt[best_gt_idx][4])
            })
            unmatched_gt.remove(best_gt_idx)

    return matches, unmatched_gt, debug_matches


def visualize_detections(image_path, ground_truth, predictions, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    height, width = img.shape[:2]

    for gt in ground_truth:
        if is_normalized_box(gt[:4]):
            x1, y1, x2, y2 = denormalize_box(gt[:4], width, height)
        else:
            x1, y1, x2, y2 = gt[:4]

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"GT:{int(gt[4])}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for pred in predictions:
        if is_normalized_box(pred[:4]):
            x1, y1, x2, y2 = denormalize_box(pred[:4], width, height)
        else:
            x1, y1, x2, y2 = pred[:4]

        conf = pred[5] if len(pred) > 5 else 0
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"Pred:{int(pred[4])}:{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(output_path, img)


def evaluate_models(models, sample_images):
    results = {}
    debug_info = []

    for model_name, model in models:
        print(f"\n[INFO] Evaluating model: {model_name}")

        metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'correct_class': 0,
            'processing_time': 0,
            'y_true': [],
            'y_pred': [],
            'class_predictions': defaultdict(Counter),
            'confidence_scores': [],
            'iou_scores': []
        }

        model_debug = {
            'name': model_name,
            'samples': []
        }

        for i, img_path in enumerate(tqdm(sample_images, desc=f"[PROGRESS] Processing with {model_name}")):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Could not read image {img_path}")
                continue

            img_height, img_width = img.shape[:2]

            label_path = get_label_path(img_path)
            ground_truth = read_yolo_label(label_path)

            start_time = time.time()
            results_model = model(img, conf=CONFIG["conf_threshold"], verbose=False)
            inference_time = time.time() - start_time
            metrics['processing_time'] += inference_time

            predictions = []
            if results_model and len(results_model) > 0 and results_model[0].boxes is not None:
                boxes = results_model[0].boxes
                for j in range(len(boxes)):
                    box = boxes[j].xyxy[0].cpu().numpy()
                    conf = boxes[j].conf[0].cpu().numpy()
                    cls = boxes[j].cls[0].cpu().numpy()
                    predictions.append([*box, cls, conf])

            if CONFIG["debug"] and i < CONFIG["debug_sample_size"]:
                sample_debug = {
                    'image_path': img_path,
                    'image_size': (img_width, img_height),
                    'ground_truth': ground_truth,
                    'predictions': predictions
                }
                model_debug['samples'].append(sample_debug)

            if len(predictions) > 0 and len(ground_truth) > 0:
                matches, unmatched_gt, debug_matches = match_predictions_with_ground_truth(
                    predictions, ground_truth, CONFIG["iou_threshold"], img_width, img_height
                )

                if CONFIG["debug"] and i < CONFIG["debug_sample_size"] and debug_matches is not None:
                    model_debug['samples'][i]['debug_matches'] = debug_matches

                metrics['true_positives'] += len(matches)
                metrics['false_positives'] += len(predictions) - len(matches)
                metrics['false_negatives'] += len(unmatched_gt)

                for match in matches:
                    pred_class = match['pred_class']
                    gt_class = match['gt_class']

                    metrics['y_true'].append(gt_class)
                    metrics['y_pred'].append(pred_class)
                    metrics['class_predictions'][gt_class][pred_class] += 1

                    if gt_class == pred_class:
                        metrics['correct_class'] += 1

                    pred_idx = match['pred_idx']
                    metrics['confidence_scores'].append((gt_class, pred_class, predictions[pred_idx][5]))
                    metrics['iou_scores'].append(match['iou'])

                for gt_idx in unmatched_gt:
                    metrics['y_true'].append(ground_truth[gt_idx][4])
                    metrics['y_pred'].append(None)

            elif len(ground_truth) > 0:
                metrics['false_negatives'] += len(ground_truth)

                for gt_box in ground_truth:
                    metrics['y_true'].append(gt_box[4])
                    metrics['y_pred'].append(None)

            elif len(predictions) > 0:
                metrics['false_positives'] += len(predictions)

        if CONFIG["debug"]:
            debug_info.append(model_debug)

        results[model_name] = calculate_overall_metrics(metrics)

        print(f"\n[INFO] Key statistics for {model_name}:")
        print(f"[INFO] - TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
        print(f"[INFO] - Detection Precision: {results[model_name]['detection_precision']:.4f}")
        print(f"[INFO] - Detection Recall: {results[model_name]['detection_recall']:.4f}")
        print(f"[INFO] - Detection F1: {results[model_name]['detection_f1']:.4f}")
        print(f"[INFO] - Class Accuracy: {results[model_name]['class_accuracy']:.4f}")
        print(f"[INFO] - Average IoU: {results[model_name]['avg_iou']:.4f}")
        print(f"[INFO] - Average Inference Time: {results[model_name]['avg_inference_time'] * 1000:.2f} ms")

    if CONFIG["debug"]:
        debug_path = os.path.join(CONFIG["output_dir"], 'debug_info.txt')
        with open(debug_path, 'w') as f:
            f.write("# Model Comparison Debug Information\n\n")
            for model_debug in debug_info:
                f.write(f"## {model_debug['name']}\n\n")

                for i, sample in enumerate(model_debug['samples']):
                    f.write(f"### Sample {i + 1}: {os.path.basename(sample['image_path'])}\n")
                    f.write(f"  Image size: {sample['image_size']}\n")
                    f.write(f"  Ground truth boxes: {len(sample['ground_truth'])}\n")

                    for j, gt in enumerate(sample['ground_truth']):
                        f.write(
                            f"    GT {j + 1}: Class {int(gt[4])}, Box: [{gt[0]:.4f}, {gt[1]:.4f}, {gt[2]:.4f}, {gt[3]:.4f}]\n")

                    f.write(f"  Predictions: {len(sample['predictions'])}\n")
                    for j, pred in enumerate(sample['predictions']):
                        f.write(
                            f"    Pred {j + 1}: Class {int(pred[4])}, Conf: {pred[5]:.4f}, Box: [{pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f}, {pred[3]:.4f}]\n")

                    if 'debug_matches' in sample:
                        f.write("\n  Matching details:\n")
                        for j, match_info in enumerate(sample['debug_matches']):
                            f.write(f"    Prediction {j + 1}:\n")
                            f.write(f"      Class: {match_info['pred_class']}, Conf: {match_info['pred_conf']:.4f}\n")
                            f.write(f"      Box: [{match_info['pred_box'][0]:.4f}, {match_info['pred_box'][1]:.4f}, "
                                    f"{match_info['pred_box'][2]:.4f}, {match_info['pred_box'][3]:.4f}]\n")
                            f.write(f"      Best IoU: {match_info['best_iou']:.4f}, Matched: {match_info['matched']}\n")
                            f.write(f"      All IoUs: {len(match_info['all_ious'])}\n")
                            for k, (gt_idx, iou, gt_class) in enumerate(
                                    sorted(match_info['all_ious'], key=lambda x: x[1], reverse=True)[:5]):
                                f.write(f"        GT {gt_idx}: Class {int(gt_class)}, IoU: {iou:.4f}\n")

                    f.write("\n")

        print(f"[INFO] Debug information saved to: {debug_path}")

    return results


def calculate_overall_metrics(metrics):
    valid_indices = [i for i, pred in enumerate(metrics['y_pred']) if pred is not None]
    y_true_valid = [metrics['y_true'][i] for i in valid_indices]
    y_pred_valid = [metrics['y_pred'][i] for i in valid_indices]

    if len(y_true_valid) > 0 and len(y_pred_valid) > 0:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_valid, y_pred_valid, labels=[0, 1], average=None, zero_division=0
        )
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
            y_true_valid, y_pred_valid, labels=[0, 1], average='weighted', zero_division=0
        )
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
    else:
        precision = recall = f1 = [0, 0]
        avg_precision = avg_recall = avg_f1 = accuracy = 0

    class_precisions = {}
    for cls in [0, 1]:
        if cls in metrics['class_predictions'] and sum(metrics['class_predictions'][cls].values()) > 0:
            if metrics['class_predictions'][cls][cls] > 0:
                class_precisions[cls] = metrics['class_predictions'][cls][cls] / sum(
                    metrics['class_predictions'][cls].values())
            else:
                class_precisions[cls] = 0
        else:
            class_precisions[cls] = 0

    map_score = sum(class_precisions.values()) / len(class_precisions) if class_precisions else 0

    if len(y_true_valid) > 0 and len(y_pred_valid) > 0:
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
    else:
        cm = np.zeros((2, 2))

    tp = metrics['true_positives']
    fp = metrics['false_positives']
    fn = metrics['false_negatives']

    det_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    det_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0

    class_acc = metrics['correct_class'] / tp if tp > 0 else 0

    avg_iou = sum(metrics['iou_scores']) / len(metrics['iou_scores']) if metrics['iou_scores'] else 0

    avg_time = metrics['processing_time'] / len(metrics['y_true']) if metrics['y_true'] else 0

    return {
        'confusion_matrix': cm,
        'class_precision': precision,
        'class_recall': recall,
        'class_f1': f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'accuracy': accuracy,
        'map_score': map_score,
        'detection_precision': det_precision,
        'detection_recall': det_recall,
        'detection_f1': det_f1,
        'class_accuracy': class_acc,
        'avg_iou': avg_iou,
        'avg_inference_time': avg_time,
        'confidence_scores': metrics['confidence_scores'],
        'class_predictions': metrics['class_predictions'],
        'raw_counts': {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    }


def visualize_results(results):
    model_names = list(results.keys())

    # 1. Bar chart of key metrics
    metrics = ['detection_precision', 'detection_recall', 'detection_f1',
               'accuracy', 'class_accuracy', 'avg_iou', 'map_score']
    metric_labels = ['Det. Precision', 'Det. Recall', 'Det. F1',
                     'Class Accuracy', 'Class Match Rate', 'Avg IoU', 'mAP']

    metrics_data = []
    for model in model_names:
        model_metrics = [results[model][metric] for metric in metrics]
        metrics_data.append(model_metrics)

    plt.figure(figsize=(14, 8))
    x = np.arange(len(metrics))
    width = 0.2
    offsets = np.linspace(-(len(model_names) - 1) / 2 * width, (len(model_names) - 1) / 2 * width,
                          len(model_names))

    for i, (model, offset) in enumerate(zip(model_names, offsets)):
        plt.bar(x + offset, metrics_data[i], width, label=model)

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison - Performance Metrics')
    plt.xticks(x, metric_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], 'metrics_comparison.png'), dpi=300)

    # 2. Confusion matrices
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]

    class_labels = ['Pollen (0)', 'Varroa (1)']

    for i, model in enumerate(model_names):
        cm = results[model]['confusion_matrix']
        row_sums = cm.sum(axis=1)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_percent = (cm.astype('float') / row_sums[:, np.newaxis]) * 100

        sns.heatmap(cm_percent, annot=cm, fmt='.0f', cmap='Blues', cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels, ax=axes[i])
        axes[i].set_title(f'{model} - Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], 'confusion_matrices.png'), dpi=300)

    # 3. Detection counts
    plt.figure(figsize=(14, 6))
    bar_width = 0.25
    index = np.arange(3)

    for i, model in enumerate(model_names):
        tp = results[model]['raw_counts']['true_positives']
        fp = results[model]['raw_counts']['false_positives']
        fn = results[model]['raw_counts']['false_negatives']
        plt.bar(index + i * bar_width, [tp, fp, fn], bar_width, label=model)

    plt.title('Detection Counts')
    plt.ylabel('Count')
    plt.xlabel('Metric')
    plt.xticks(index + bar_width, ['True Positives', 'False Positives', 'False Negatives'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], 'detection_counts.png'), dpi=300)

    # 4. Inference time comparison
    times = [results[model]['avg_inference_time'] * 1000 for model in model_names]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, times, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Average Inference Time (ms)')
    plt.title('Model Comparison - Inference Speed')
    plt.xticks(rotation=45)
    for i, time in enumerate(times):
        plt.text(i, time + 1, f'{time:.2f} ms', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], 'inference_times.png'), dpi=300)

    # 5. Confidence score distribution
    plt.figure(figsize=(14, 7))

    for i, model in enumerate(model_names):
        plt.subplot(1, len(model_names), i + 1)

        if 'confidence_scores' not in results[model]:
            print(
                f"[WARNING] 'confidence_scores' key not found for {model}. Skipping confidence distribution visualization.")
            plt.text(0.5, 0.5, "No confidence data",
                     horizontalalignment='center', verticalalignment='center')
            plt.title(f'{model} - No Confidence Data')
            plt.axis('off')
            continue

        confidence_data = results[model]['confidence_scores']

        if confidence_data:
            correct_conf = [conf for gt, pred, conf in confidence_data if gt == pred]
            incorrect_conf = [conf for gt, pred, conf in confidence_data if gt != pred]

            if correct_conf or incorrect_conf:
                bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
                plt.hist([correct_conf, incorrect_conf], bins=bins,
                         label=['Correct', 'Incorrect'], alpha=0.7)
                plt.title(f'{model} - Confidence Distribution')
                plt.xlabel('Confidence Score')
                plt.ylabel('Count')
                plt.legend()
            else:
                plt.text(0.5, 0.5, "No classification data",
                         horizontalalignment='center', verticalalignment='center')
                plt.title(f'{model} - No Class Matches')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, "No confidence data",
                     horizontalalignment='center', verticalalignment='center')
            plt.title(f'{model} - No Matches')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], 'confidence_distribution.png'), dpi=300)

    # 6. Class Distribution in True Positives
    plt.figure(figsize=(14, 7))

    for i, model in enumerate(model_names):
        plt.subplot(1, len(model_names), i + 1)

        if 'class_predictions' not in results[model]:
            print(
                f"[WARNING] 'class_predictions' key not found for {model}. Skipping class distribution visualization.")
            plt.text(0.5, 0.5, "No class data",
                     horizontalalignment='center', verticalalignment='center')
            plt.title(f'{model} - No Class Data')
            plt.axis('off')
            continue

        class_data = results[model]['class_predictions']

        if class_data:
            bee_correct = class_data.get(0, {}).get(0, 0)
            bee_incorrect = class_data.get(0, {}).get(1, 0) if 1 in class_data.get(0, {}) else 0
            wasp_correct = class_data.get(1, {}).get(1, 0)
            wasp_incorrect = class_data.get(1, {}).get(0, 0) if 0 in class_data.get(1, {}) else 0

            data = [bee_correct, bee_incorrect, wasp_correct, wasp_incorrect]
            labels = ['Pollen→Pollen', 'Pollen→Varroa', 'Varroa→Varroa', 'Varroa→Pollen']
            colors = ['green', 'red', 'blue', 'orange']

            if sum(data) > 0:
                plt.pie(data, labels=labels, autopct='%1.1f%%', colors=colors)
                plt.title(f'{model}')
            else:
                plt.text(0.5, 0.5, "No class data",
                         horizontalalignment='center', verticalalignment='center')
                plt.title(f'{model} - No Class Data')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, "No class data",
                     horizontalalignment='center', verticalalignment='center')
            plt.title(f'{model} - No Class Data')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], 'class_distribution.png'), dpi=300)


def save_summary_report(results, sample_images):
    report_path = os.path.join(CONFIG["output_dir"], 'model_comparison_report.txt')

    with open(report_path, 'w') as f:
        f.write("# YOLO Model Comparison Report\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of test samples: {len(sample_images)}\n")
        f.write(f"Confidence threshold: {CONFIG['conf_threshold']}\n")
        f.write(f"IoU threshold for matching: {CONFIG['iou_threshold']}\n\n")

        f.write("## Models Evaluated\n\n")
        for i, model_path in enumerate(CONFIG["model_paths"]):
            f.write(f"{i + 1}. {model_path}\n")
        f.write("\n")

        f.write("## Overall Results Summary\n\n")

        metrics_to_show = [
            ('Detection Precision', 'detection_precision'),
            ('Detection Recall', 'detection_recall'),
            ('Detection F1-Score', 'detection_f1'),
            ('Classification Accuracy', 'accuracy'),
            ('Class Match Rate', 'class_accuracy'),
            ('mAP Score', 'map_score'),
            ('Average IoU', 'avg_iou'),
            ('Avg Inference Time (ms)', 'avg_inference_time')
        ]

        f.write("| Metric | " + " | ".join(results.keys()) + " |\n")
        f.write("|" + "-" * 10 + "|" + "".join(["-" * 12 + "|" for _ in results]) + "\n")

        for metric_name, metric_key in metrics_to_show:
            row = f"| {metric_name} | "
            for model in results:
                value = results[model][metric_key]
                if metric_key == 'avg_inference_time':
                    value = value * 1000
                    row += f"{value:.2f} ms | "
                else:
                    row += f"{value:.4f} | "
            f.write(row + "\n")

        f.write("\n")

        for model_name in results:
            f.write(f"## Detailed Results for {model_name}\n\n")

            f.write("### Confusion Matrix\n\n")
            cm = results[model_name]['confusion_matrix']
            f.write("```\n")
            f.write(f"             | Pred Pollen | Pred Varroa |\n")
            f.write(f"True Pollen (0) | {int(cm[0, 0]):9d} | {int(cm[0, 1]):9d} |\n")
            f.write(f"True Varroa (1)| {int(cm[1, 0]):9d} | {int(cm[1, 1]):9d} |\n")
            f.write("```\n\n")

            f.write("### Class-specific Metrics\n\n")
            f.write("```\n")
            f.write(f"           | Pollen (0)  | Varroa (1) |\n")
            f.write(
                f"Precision  | {results[model_name]['class_precision'][0]:.4f} | {results[model_name]['class_precision'][1]:.4f} |\n")
            f.write(
                f"Recall     | {results[model_name]['class_recall'][0]:.4f} | {results[model_name]['class_recall'][1]:.4f} |\n")
            f.write(
                f"F1-Score   | {results[model_name]['class_f1'][0]:.4f} | {results[model_name]['class_f1'][1]:.4f} |\n")
            f.write("```\n\n")

            f.write("### Detection Counts\n\n")
            counts = results[model_name]['raw_counts']
            f.write(f"- True Positives: {counts['true_positives']}\n")
            f.write(f"- False Positives: {counts['false_positives']}\n")
            f.write(f"- False Negatives: {counts['false_negatives']}\n\n")

        f.write("## Conclusion\n\n")

        if results:
            best_f1_model = max(results.items(), key=lambda x: x[1]['detection_f1'])[0]
            best_accuracy_model = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
            best_iou_model = max(results.items(), key=lambda x: x[1]['avg_iou'])[0]
            fastest_model = min(results.items(), key=lambda x: x[1]['avg_inference_time'])[0]

            f.write(f"- Best model for overall detection (F1): {best_f1_model}\n")
            f.write(f"- Best model for classification accuracy: {best_accuracy_model}\n")
            f.write(f"- Best model for detection precision (IoU): {best_iou_model}\n")
            f.write(f"- Fastest model: {fastest_model}\n\n")
        else:
            f.write("No valid results available for comparison.\n\n")

        if CONFIG["debug"]:
            f.write("## Debug Notes\n\n")
            f.write("This evaluation was run with DEBUG=True. Check the debug_info.txt file for detailed information.\n")
            f.write("Common issues with model evaluation:\n")
            f.write("1. Coordinate format mismatch: YOLO uses normalized coordinates, but model may output pixel coordinates\n")
            f.write("2. IoU threshold too high: If boxes don't overlap enough, they won't match\n")
            f.write("3. Label paths incorrect: Make sure the ground truth label files are correctly paired with images\n")
            f.write("4. Class ID mismatch: Ensure that class IDs match between predictions and ground truth\n\n")

        f.write("Please refer to the generated visualization files for more detailed comparisons.\n")

    print(f"[INFO] Report saved to: {report_path}")


def main():
    print("[INFO] Starting YOLO Model Comparison")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    models = load_models()
    if not models:
        print("[ERROR] No models could be loaded. Exiting.")
        return

    print(f"[INFO] Selecting {CONFIG['num_samples']} random images for evaluation")
    sample_images = get_random_samples(CONFIG['num_samples'])
    print(f"[INFO] Selected {len(sample_images)} images")

    results = evaluate_models(models, sample_images)

    any_true_positives = False
    for model_name in results:
        if results[model_name]['raw_counts']['true_positives'] > 0:
            any_true_positives = True
            break

    if not any_true_positives:
        print("\n")
        print("=" * 80)
        print("[WARNING] No true positives found for any model!")
        print("[INFO] This usually indicates one of these problems:")
        print("[INFO] 1. Coordinate system mismatch: YOLO labels use normalized coordinates (0-1)")
        print("[INFO] 2. IoU threshold too high: Try lowering CONFIG['iou_threshold']")
        print("[INFO] 3. Confidence threshold too high: Try lowering CONFIG['conf_threshold']")
        print("[INFO] 4. Path to labels incorrect")
        print("[INFO] Check the debug_info.txt file for more detailed information.")
        print("=" * 80)
        print("\n")

    print("[INFO] Generating visualizations...")
    try:
        visualize_results(results)
        print("[INFO] Visualizations completed successfully.")
    except Exception as e:
        print(f"[ERROR] Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

    print("[INFO] Generating summary report...")
    try:
        save_summary_report(results, sample_images)
        print("[INFO] Summary report completed successfully.")
    except Exception as e:
        print(f"[ERROR] Error generating summary report: {e}")
        import traceback
        traceback.print_exc()

    print(f"[INFO] All results saved to: {os.path.abspath(CONFIG['output_dir'])}")
    print("[INFO] Comparison completed!")


if __name__ == "__main__":
    random.seed(42)

    try:
        main()
    except Exception as e:
        print(f"[ERROR] Main execution error: {e}")
        import traceback
        traceback.print_exc()