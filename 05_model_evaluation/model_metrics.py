import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from ultralytics import YOLO
from torchvision.ops import box_iou
import json
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "data_dir": os.path.join(CURRENT_DIR, "..", "03_custom_dataset", "bee_vs_wasp_yolo"),
    "output_dir": os.path.join(CURRENT_DIR, "..", "03_custom_dataset", "bee_vs_wasp_yolo", "yolo_dataset"),
    "model_path": os.path.join(CURRENT_DIR, "..", "04_model_training", "runs", "detect", "bee_wasp_model_75_640",
                               "weights", "best.pt"),
    "plot_style": 'seaborn-v0_8-darkgrid',
    "classes": ["bee", "wasp"],
    "analysis_output_dir": os.path.join(CURRENT_DIR, "analysis"),
    "conf_threshold": 0.4,
    "iou_threshold": 0.5,
    "sample_size": 100,
    "verbose": False,
    "dpi": 300,
}


def analyze_object_sizes():
    label_dir = os.path.join(CONFIG["output_dir"], "train", "labels")
    class_widths = [[] for _ in range(len(CONFIG["classes"]))]
    class_heights = [[] for _ in range(len(CONFIG["classes"]))]

    for label_file in os.listdir(label_dir):
        with open(os.path.join(label_dir, label_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    _, x_center, y_center, w, h = map(float, parts)
                    if class_id < len(CONFIG["classes"]):
                        class_widths[class_id].append(w)
                        class_heights[class_id].append(h)

    plt.figure(figsize=(14, 10))
    plt.style.use(CONFIG["plot_style"])

    plt.subplot(2, 2, 1)
    for i, widths in enumerate(class_widths):
        sns.kdeplot(widths, label=CONFIG["classes"][i])
    plt.title('Object Width Distribution by Class')
    plt.xlabel('Normalized Width')
    plt.legend()

    plt.subplot(2, 2, 2)
    for i, heights in enumerate(class_heights):
        sns.kdeplot(heights, label=CONFIG["classes"][i])
    plt.title('Object Height Distribution by Class')
    plt.xlabel('Normalized Height')
    plt.legend()

    plt.subplot(2, 2, 3)
    all_widths = [w for class_w in class_widths for w in class_w]
    sns.histplot(all_widths, bins=50, kde=True)
    plt.title('Width Distribution for All Classes')
    plt.xlabel('Normalized Width')

    plt.subplot(2, 2, 4)
    all_heights = [h for class_h in class_heights for h in class_h]
    sns.histplot(all_heights, bins=50, kde=True)
    plt.title('Height Distribution for All Classes')
    plt.xlabel('Normalized Height')

    plt.tight_layout()
    save_plot('object_size_distribution.png')

    plt.figure(figsize=(12, 8))
    for i in range(len(CONFIG["classes"])):
        if class_widths[i] and class_heights[i]:
            plt.scatter(class_widths[i], class_heights[i], label=CONFIG["classes"][i], alpha=0.6)
    plt.title('Object Size Distribution (Width vs Height)')
    plt.xlabel('Normalized Width')
    plt.ylabel('Normalized Height')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot('object_size_scatter.png')


def calculate_iou_distribution(model):
    test_img_dir = os.path.join(CONFIG["output_dir"], "test", "images")
    test_label_dir = os.path.join(CONFIG["output_dir"], "test", "labels")
    ious = []
    class_ious = [[] for _ in range(len(CONFIG["classes"]))]

    for img_file in tqdm(os.listdir(test_img_dir), desc="Calculating IoU"):
        img_path = os.path.join(test_img_dir, img_file)
        label_path = os.path.join(test_label_dir, os.path.splitext(img_file)[0] + ".txt")

        gt_boxes = []
        gt_classes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        _, x_center, y_center, w, h = map(float, parts)
                        x1 = (x_center - w / 2)
                        y1 = (y_center - h / 2)
                        x2 = (x_center + w / 2)
                        y2 = (y_center + h / 2)
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_classes.append(class_id)

        results = model.predict(img_path, conf=CONFIG["conf_threshold"], verbose=CONFIG["verbose"])
        pred_boxes = results[0].boxes.xyxyn.cpu().numpy() if len(results[0].boxes) > 0 else []
        pred_classes = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes) > 0 else []

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            gt_tensor = torch.tensor(gt_boxes)
            pred_tensor = torch.tensor(pred_boxes)

            if len(gt_tensor.shape) == 1:
                gt_tensor = gt_tensor.unsqueeze(0)
            if len(pred_tensor.shape) == 1:
                pred_tensor = pred_tensor.unsqueeze(0)

            iou_matrix = box_iou(gt_tensor, pred_tensor)

            if iou_matrix.numel() > 0:
                for i, class_id in enumerate(gt_classes):
                    if i < iou_matrix.shape[0]:
                        class_pred_indices = [j for j, c in enumerate(pred_classes) if int(c) == class_id]
                        if class_pred_indices:
                            class_ious_row = [iou_matrix[i, j].item() if j < iou_matrix.shape[1] else 0
                                              for j in class_pred_indices]
                            max_iou = max(class_ious_row) if class_ious_row else 0
                            if max_iou > 0:
                                ious.append(max_iou)
                                if class_id < len(class_ious):
                                    class_ious[class_id].append(max_iou)

    if ious:
        plt.figure(figsize=(10, 6))
        sns.histplot(ious, bins=50, kde=True)
        plt.title(f'IoU Distribution (Mean: {np.mean(ious):.2f})')
        plt.xlabel('Intersection over Union (IoU)')
        save_plot('iou_distribution.png')

        plt.figure(figsize=(12, 6))
        for i, class_iou in enumerate(class_ious):
            if class_iou:
                sns.kdeplot(class_iou, label=f"{CONFIG['classes'][i]} (Mean: {np.mean(class_iou):.2f})")
        plt.title('IoU Distribution by Class')
        plt.xlabel('IoU')
        plt.legend()
        save_plot('class_iou_distribution.png')
    else:
        print("[WARNING] Could not calculate IoU values. No matching boxes found.")


def analyze_error_types(model):
    test_img_dir = os.path.join(CONFIG["output_dir"], "test", "images")
    test_label_dir = os.path.join(CONFIG["output_dir"], "test", "labels")

    error_counts = {'FP': 0, 'FN': 0, 'Localization': 0}
    class_error_counts = [{
        'FP': 0, 'FN': 0, 'Localization': 0, 'TP': 0
    } for _ in range(len(CONFIG["classes"]))]

    iou_threshold = CONFIG["iou_threshold"]

    for img_file in tqdm(os.listdir(test_img_dir), desc="Error Analysis"):
        img_path = os.path.join(test_img_dir, img_file)
        label_path = os.path.join(test_label_dir, os.path.splitext(img_file)[0] + ".txt")

        gt_boxes = []
        gt_classes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        _, x_center, y_center, w, h = map(float, parts)
                        x1 = (x_center - w / 2)
                        y1 = (y_center - h / 2)
                        x2 = (x_center + w / 2)
                        y2 = (y_center + h / 2)
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_classes.append(class_id)

        results = model.predict(img_path, conf=CONFIG["conf_threshold"], verbose=CONFIG["verbose"])
        pred_boxes = results[0].boxes.xyxyn.cpu().numpy() if len(results[0].boxes) > 0 else []
        pred_classes = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes) > 0 else []

        if len(gt_boxes) == 0:
            error_counts['FP'] += len(pred_boxes)
            for pred_class in pred_classes:
                class_id = int(pred_class)
                if class_id < len(class_error_counts):
                    class_error_counts[class_id]['FP'] += 1
        else:
            matched = set()
            matched_classes = [False] * len(gt_boxes)

            if len(pred_boxes) > 0:
                gt_tensor = torch.tensor(gt_boxes)
                pred_tensor = torch.tensor(pred_boxes)

                if len(gt_tensor.shape) == 1:
                    gt_tensor = gt_tensor.unsqueeze(0)
                if len(pred_tensor.shape) == 1:
                    pred_tensor = pred_tensor.unsqueeze(0)

                iou_matrix = box_iou(gt_tensor, pred_tensor)

                for i in range(len(gt_boxes)):
                    class_id = gt_classes[i]
                    if iou_matrix.shape[1] > 0:
                        max_iou, max_iou_idx = torch.max(iou_matrix[i], dim=0)
                        max_iou_val = max_iou.item()
                        max_idx = max_iou_idx.item()

                        if max_iou_val >= iou_threshold:
                            if max_idx < len(pred_classes) and int(pred_classes[max_idx]) == class_id:
                                matched.add(max_idx)
                                matched_classes[i] = True
                                if class_id < len(class_error_counts):
                                    class_error_counts[class_id]['TP'] += 1
                            else:
                                error_counts['FN'] += 1
                                if class_id < len(class_error_counts):
                                    class_error_counts[class_id]['FN'] += 1
                        elif 0.1 < max_iou_val < iou_threshold:
                            error_counts['Localization'] += 1
                            if class_id < len(class_error_counts):
                                class_error_counts[class_id]['Localization'] += 1
                        else:
                            error_counts['FN'] += 1
                            if class_id < len(class_error_counts):
                                class_error_counts[class_id]['FN'] += 1
                    else:
                        error_counts['FN'] += 1
                        if class_id < len(class_error_counts):
                            class_error_counts[class_id]['FN'] += 1

                for j in range(len(pred_boxes)):
                    if j not in matched:
                        error_counts['FP'] += 1
                        class_id = int(pred_classes[j]) if j < len(pred_classes) else 0
                        if class_id < len(class_error_counts):
                            class_error_counts[class_id]['FP'] += 1
            else:
                error_counts['FN'] += len(gt_boxes)
                for i, class_id in enumerate(gt_classes):
                    if class_id < len(class_error_counts):
                        class_error_counts[class_id]['FN'] += 1

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(error_counts.keys(), error_counts.values(), color=['red', 'blue', 'green'])
    plt.title('General Error Type Distribution')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    x = np.arange(len(CONFIG["classes"]))
    width = 0.2

    fp_vals = [class_error_counts[i]['FP'] for i in range(len(CONFIG["classes"]))]
    fn_vals = [class_error_counts[i]['FN'] for i in range(len(CONFIG["classes"]))]
    loc_vals = [class_error_counts[i]['Localization'] for i in range(len(CONFIG["classes"]))]
    tp_vals = [class_error_counts[i]['TP'] for i in range(len(CONFIG["classes"]))]

    plt.bar(x - 1.5 * width, fp_vals, width, label='FP', color='red')
    plt.bar(x - 0.5 * width, fn_vals, width, label='FN', color='blue')
    plt.bar(x + 0.5 * width, loc_vals, width, label='Localization', color='green')
    plt.bar(x + 1.5 * width, tp_vals, width, label='TP', color='gray')

    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Error Distribution by Class')
    plt.xticks(x, CONFIG["classes"])
    plt.legend()

    plt.tight_layout()
    save_plot('error_type_analysis.png')

    create_confusion_matrix(model)

    return class_error_counts


def create_confusion_matrix(model):
    test_img_dir = os.path.join(CONFIG["output_dir"], "test", "images")
    test_label_dir = os.path.join(CONFIG["output_dir"], "test", "labels")

    y_true = []
    y_pred = []
    iou_threshold = CONFIG["iou_threshold"]

    for img_file in tqdm(os.listdir(test_img_dir), desc="Creating Confusion Matrix"):
        img_path = os.path.join(test_img_dir, img_file)
        label_path = os.path.join(test_label_dir, os.path.splitext(img_file)[0] + ".txt")

        gt_classes = []
        gt_boxes = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id < len(CONFIG["classes"]):
                            gt_classes.append(class_id)
                            _, x_center, y_center, w, h = map(float, parts)
                            x1 = (x_center - w / 2)
                            y1 = (y_center - h / 2)
                            x2 = (x_center + w / 2)
                            y2 = (y_center + h / 2)
                            gt_boxes.append([x1, y1, x2, y2])

        results = model.predict(img_path, conf=CONFIG["conf_threshold"], verbose=CONFIG["verbose"])
        pred_classes = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes) > 0 else []
        pred_boxes = results[0].boxes.xyxyn.cpu().numpy() if len(results[0].boxes) > 0 else []

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            gt_tensor = torch.tensor(gt_boxes)
            pred_tensor = torch.tensor(pred_boxes)

            if len(gt_tensor.shape) == 1:
                gt_tensor = gt_tensor.unsqueeze(0)
            if len(pred_tensor.shape) == 1:
                pred_tensor = pred_tensor.unsqueeze(0)

            iou_matrix = box_iou(gt_tensor, pred_tensor)

            for i, true_class in enumerate(gt_classes):
                if i < iou_matrix.shape[0]:
                    max_iou, max_idx = torch.max(iou_matrix[i], dim=0)
                    if max_iou >= iou_threshold:
                        if max_idx < len(pred_classes):
                            pred_class = int(pred_classes[max_idx])
                            y_true.append(true_class)
                            y_pred.append(pred_class)
                    else:
                        y_true.append(true_class)
                        y_pred.append(-1)  # Not detected

        if len(gt_boxes) == 0 and len(pred_boxes) > 0:
            for pred_class in pred_classes:
                y_true.append(-1)
                y_pred.append(int(pred_class))

    if y_true and y_pred:
        all_classes = sorted(set(y_true + y_pred))
        valid_classes = [c for c in all_classes if c >= 0 and c < len(CONFIG["classes"])]

        cm = confusion_matrix(
            [y for y in y_true if y in valid_classes],
            [y for y, t in zip(y_pred, y_true) if t in valid_classes],
            labels=valid_classes
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[CONFIG["classes"][i] for i in valid_classes],
                    yticklabels=[CONFIG["classes"][i] for i in valid_classes])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        save_plot('confusion_matrix.png')

        if np.sum(cm) > 0:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=[CONFIG["classes"][i] for i in valid_classes],
                        yticklabels=[CONFIG["classes"][i] for i in valid_classes])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Normalized Confusion Matrix')
            save_plot('normalized_confusion_matrix.png')
    else:
        print("[WARNING] Not enough data for confusion matrix.")


def plot_precision_recall_curve(model):
    test_img_dir = os.path.join(CONFIG["output_dir"], "test", "images")
    test_label_dir = os.path.join(CONFIG["output_dir"], "test", "labels")

    class_y_true = [[] for _ in range(len(CONFIG["classes"]))]
    class_y_scores = [[] for _ in range(len(CONFIG["classes"]))]

    for img_file in tqdm(os.listdir(test_img_dir), desc="Creating PR Curve"):
        img_path = os.path.join(test_img_dir, img_file)
        label_path = os.path.join(test_label_dir, os.path.splitext(img_file)[0] + ".txt")

        gt_classes = set()
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id < len(CONFIG["classes"]):
                            gt_classes.add(class_id)

        results = model.predict(img_path, verbose=CONFIG["verbose"])

        for class_id in range(len(CONFIG["classes"])):
            class_y_true[class_id].append(class_id in gt_classes)

            class_scores = []
            if len(results[0].boxes) > 0:
                for i, cls in enumerate(results[0].boxes.cls.cpu().numpy()):
                    if int(cls) == class_id:
                        class_scores.append(results[0].boxes.conf[i].cpu().item())

            class_y_scores[class_id].append(max(class_scores) if class_scores else 0)

    plt.figure(figsize=(12, 8))

    for i in range(len(CONFIG["classes"])):
        precision, recall, _ = precision_recall_curve(class_y_true[i], class_y_scores[i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{CONFIG["classes"][i]} (AUC={pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves by Class')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot('precision_recall_curves.png')


def save_plot(filename):
    os.makedirs(CONFIG["analysis_output_dir"], exist_ok=True)
    output_path = os.path.join(CONFIG["analysis_output_dir"], filename)
    plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight')
    print(f"[INFO] Plot saved: {output_path}")


def analyze_class_distribution():
    train_label_dir = os.path.join(CONFIG["output_dir"], "train", "labels")
    val_label_dir = os.path.join(CONFIG["output_dir"], "val", "labels")
    test_label_dir = os.path.join(CONFIG["output_dir"], "test", "labels")

    counts = {
        'train': {class_name: 0 for class_name in CONFIG["classes"]},
        'val': {class_name: 0 for class_name in CONFIG["classes"]},
        'test': {class_name: 0 for class_name in CONFIG["classes"]}
    }

    for label_file in os.listdir(train_label_dir):
        with open(os.path.join(train_label_dir, label_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id < len(CONFIG["classes"]):
                        counts['train'][CONFIG["classes"][class_id]] += 1

    for label_file in os.listdir(val_label_dir):
        with open(os.path.join(val_label_dir, label_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id < len(CONFIG["classes"]):
                        counts['val'][CONFIG["classes"][class_id]] += 1

    for label_file in os.listdir(test_label_dir):
        with open(os.path.join(test_label_dir, label_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id < len(CONFIG["classes"]):
                        counts['test'][CONFIG["classes"][class_id]] += 1

    plt.figure(figsize=(12, 8))

    width = 0.25
    x = np.arange(len(CONFIG["classes"]))

    plt.bar(x - width, [counts['train'][c] for c in CONFIG["classes"]], width, label='Training', color='blue')
    plt.bar(x, [counts['val'][c] for c in CONFIG["classes"]], width, label='Validation', color='green')
    plt.bar(x + width, [counts['test'][c] for c in CONFIG["classes"]], width, label='Test', color='red')

    plt.xlabel('Class')
    plt.ylabel('Object Count')
    plt.title('Dataset Class Distribution')
    plt.xticks(x, CONFIG["classes"])
    plt.legend()

    for i, cls in enumerate(CONFIG["classes"]):
        total = counts['train'][cls] + counts['val'][cls] + counts['test'][cls]
        plt.text(i, max([counts['train'][cls], counts['val'][cls], counts['test'][cls]]) + 5,
                 f'Total: {total}', ha='center')

    save_plot('class_distribution.png')

    return counts


def export_model_performance_metrics(error_counts):
    class_metrics = {}

    for i, class_name in enumerate(CONFIG["classes"]):
        tp = error_counts[i]['TP']
        fp = error_counts[i]['FP']
        fn = error_counts[i]['FN']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'localization_errors': error_counts[i]['Localization']
        }

    total_tp = sum(error_counts[i]['TP'] for i in range(len(CONFIG["classes"])))
    total_fp = sum(error_counts[i]['FP'] for i in range(len(CONFIG["classes"])))
    total_fn = sum(error_counts[i]['FN'] for i in range(len(CONFIG["classes"])))

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (
                                                                                                total_precision + total_recall) > 0 else 0

    metrics = {
        'model_name': os.path.basename(CONFIG["model_path"]).replace('.pt', ''),
        'date': time.strftime('%Y-%m-%d'),
        'overall_metrics': {
            'precision': total_precision,
            'recall': total_recall,
            'f1_score': total_f1,
            'mAP': None,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        },
        'class_metrics': class_metrics
    }

    metrics_path = os.path.join(CONFIG["analysis_output_dir"], "model_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Model metrics saved to: {metrics_path}")

    return metrics


def main():
    try:
        os.makedirs(CONFIG["analysis_output_dir"], exist_ok=True)

        print(f"[INFO] Loading model: {CONFIG['model_path']}")
        model = YOLO(CONFIG["model_path"])

        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        os.environ["ULTRALYTICS_SILENT"] = "True"

        print("[INFO] Analyzing dataset distribution...")
        analyze_class_distribution()

        print("[INFO] Analyzing object sizes...")
        analyze_object_sizes()

        print("[INFO] Performing error analysis...")
        error_counts = analyze_error_types(model)

        print("[INFO] Calculating IoU distribution...")
        calculate_iou_distribution(model)

        print("[INFO] Creating Precision-Recall curve...")
        plot_precision_recall_curve(model)

        print("[INFO] Calculating model performance metrics...")
        export_model_performance_metrics(error_counts)

        print(f"[INFO] Analysis complete! Results are in the '{CONFIG['analysis_output_dir']}' directory.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()