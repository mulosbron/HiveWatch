import os
import cv2
import numpy as np
import glob
import traceback
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "input_dir": os.path.join(CURRENT_DIR, "source_images"),
    "output_dir": os.path.join(CURRENT_DIR, "ensemble_results"),
    "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'},
    "bee_ensemble": {
        "model_paths": [
            os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "bee_wasp_model_50_320_pre", "weights", "best.pt"),
            os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "bee_wasp_model_50_448", "weights",
                         "best.pt"),
            os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "bee_wasp_model_50_512", "weights",
                         "best.pt"),
            os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "bee_wasp_model_50_640_pre", "weights", "best.pt")

        ],
        "conf_threshold": 0.51,
        "iou_threshold": 0.15,
        "ensemble_method": "weighted_avg",
        "auto_weights": True,
        "model_performances": {
            "bee_wasp_model_50_320_pre": 0.57619,
            "bee_wasp_model_50_448": 0.58067,
            "bee_wasp_model_50_512": 0.60746,
            "bee_wasp_model_50_640_pre": 0.52795,
        }
    },
    "pv_ensemble": {
        "model_paths": [
            os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "pollen_varroa_model_50_320", "weights", "best.pt"),
            os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "pollen_varroa_model_50_320_pre", "weights", "best.pt"),
            os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "pollen_varroa_model_50_640_pre", "weights", "best.pt"),
            os.path.join(CURRENT_DIR, "..", "03_model_training", "runs", "detect", "pollen_varroa_model_50_448_pre", "weights", "best.pt")
        ],
        "conf_threshold": 0.51,
        "iou_threshold": 0.15,
        "ensemble_method": "weighted_avg",
        "auto_weights": True,
        "model_performances": {
            "pollen_varroa_model_50_320_pre": 0.48551,
            "pollen_varroa_model_50_320": 0.45373,
            "pollen_varroa_model_50_448_pre": 0.49117,
            "pollen_varroa_model_50_640_pre": 0.47005,
        }
    }
}


class EnsembleDetector:
    def __init__(self,
                 model_paths,
                 conf_threshold,
                 iou_threshold,
                 ensemble_method='weighted_avg',
                 auto_weights=False,
                 model_performances=None):
        """
        Args:
            model_paths (list[str]): .pt file paths
            conf_threshold (float): minimum confidence
            iou_threshold (float): IOU threshold for merging
            ensemble_method (str): ‘weighted_avg’ or ‘max’
            auto_weights (bool): If True, use model_performances; otherwise, use equal
            model_performances (dict): {model_name: performance}
        """
        self.models = []
        self.conf_thresh = conf_threshold
        self.iou_thresh = iou_threshold
        self.ensemble_method = ensemble_method
        self.model_performances = model_performances or {}
        self.weights = []

        for path in model_paths:
            if path.lower().endswith('.pt'):
                self.models.append(YOLO(path))
        if not self.models:
            raise ValueError("No valid model files loaded!")

        # set weights
        n = len(self.models)
        if auto_weights and ensemble_method == 'weighted_avg' and self.model_performances:
            raw = []
            avg = sum(self.model_performances.values()) / len(self.model_performances)
            for m in self.models:
                # yol dizisinden model adı çıkar
                name = os.path.basename(os.path.dirname(os.path.dirname(m.pt_path)))
                raw.append(self.model_performances.get(name, avg))
            s = sum(raw) or 1.0
            self.weights = [r / s for r in raw]
        else:
            self.weights = [1.0 / n] * n

        self.class_names = self.models[0].names

    def _iou(self, b1, b2):
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2
        xi1, yi1 = max(x11, x21), max(y11, y21)
        xi2, yi2 = min(x12, x22), min(y12, y22)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a1 = (x12 - x11) * (y12 - y11)
        a2 = (x22 - x21) * (y22 - y21)
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0

    def detect(self, img):
        preds = []
        for model, w in zip(self.models, self.weights):
            res = model(img, conf=self.conf_thresh, verbose=False)[0]
            for *box, conf, cls in res.boxes.data.cpu().numpy():
                preds.append({
                    'bbox': [float(b) for b in box],
                    'confidence': float(conf),
                    'class': int(cls),
                    'weight': w
                })
        return self._merge(preds)

    def _merge(self, preds):
        if not preds:
            return []
        preds.sort(key=lambda x: x['confidence'], reverse=True)
        merged = []
        used = [False] * len(preds)
        for i, p in enumerate(preds):
            if used[i]:
                continue
            group = [p]
            used[i] = True
            for j in range(i + 1, len(preds)):
                if not used[j] and self._iou(p['bbox'], preds[j]['bbox']) > self.iou_thresh:
                    group.append(preds[j])
                    used[j] = True
            if self.ensemble_method == 'weighted_avg':
                total_w = sum(d['weight'] for d in group)
                bbox = sum(np.array(d['bbox']) * d['weight'] for d in group) / total_w
                conf = sum(d['confidence'] * d['weight'] for d in group) / total_w
                votes = {}
                for d in group:
                    votes[d['class']] = votes.get(d['class'], 0) + d['weight']
                cls = max(votes, key=votes.get)
                merged.append({'bbox': bbox.tolist(), 'confidence': conf, 'class': cls})
            else:  # 'max'
                best = max(group, key=lambda x: x['confidence'])
                merged.append(best)
        return merged


def process_images():
    bcfg = CONFIG["bee_ensemble"]
    pcfg = CONFIG["pv_ensemble"]
    bee_det = EnsembleDetector(
        model_paths=bcfg["model_paths"],
        conf_threshold=bcfg["conf_threshold"],
        iou_threshold=bcfg["iou_threshold"],
        ensemble_method=bcfg["ensemble_method"],
        auto_weights=bcfg["auto_weights"],
        model_performances=bcfg["model_performances"]
    )
    pv_det = EnsembleDetector(
        model_paths=pcfg["model_paths"],
        conf_threshold=pcfg["conf_threshold"],
        iou_threshold=pcfg["iou_threshold"],
        ensemble_method=pcfg["ensemble_method"],
        auto_weights=pcfg["auto_weights"],
        model_performances=pcfg["model_performances"]
    )

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    imgs = []
    for ext in CONFIG["image_extensions"]:
        imgs += glob.glob(os.path.join(CONFIG["input_dir"], f"*{ext}"))
        imgs += glob.glob(os.path.join(CONFIG["input_dir"], f"*{ext.upper()}"))
    imgs = sorted(set(imgs))
    print(f"[INFO] Found {len(imgs)} images")

    bee_id = next((idx for idx, name in bee_det.class_names.items() if name.lower() == "bee"), None)
    wasp_id = next((idx for idx, name in bee_det.class_names.items() if name.lower() == "wasp"), None)

    for idx, path in enumerate(imgs, 1):
        try:
            print(f"[PROGRESS] {idx}/{len(imgs)} {os.path.basename(path)}")
            img = cv2.imread(path)
            if img is None:
                print(f"[ERROR] Cannot read {path}")
                continue

            out = img.copy()
            preds = bee_det.detect(img)

            for p in preds:
                x1, y1, x2, y2 = map(int, p['bbox'])
                cls, conf = p['class'], p['confidence']

                if cls == bee_id:
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    lbl = f"bee {conf:.2f}"
                    tw, th = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
                    cv2.putText(out, lbl, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        pv_preds = pv_det.detect(crop)
                        for pp in pv_preds:
                            px1, py1, px2, py2 = map(int, pp['bbox'])
                            gx1, gy1 = x1 + px1, y1 + py1
                            gx2, gy2 = x1 + px2, y1 + py2
                            cname = pv_det.class_names.get(pp['class'], str(pp['class']))
                            col = (0, 0, 255) if cname.lower() == "varroa" else (255, 0, 0)
                            cv2.rectangle(out, (gx1, gy1), (gx2, gy2), col, 2)
                            ov = f"{cname} {pp['confidence']:.2f}"
                            tw2, th2 = cv2.getTextSize(ov, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            cv2.rectangle(out, (gx1, gy1 - th2 - 6), (gx1 + tw2, gy1), col, -1)
                            cv2.putText(out, ov, (gx1, gy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                elif cls == wasp_id:
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    lbl = f"wasp {conf:.2f}"
                    tw, th = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), (0, 0, 255), -1)
                    cv2.putText(out, lbl, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out_path = os.path.join(CONFIG["output_dir"], f"{os.path.basename(path)}")
            cv2.imwrite(out_path, out)

        except Exception as e:
            print(f"[ERROR] {os.path.basename(path)}: {e}")
            traceback.print_exc()

    print("[INFO] Done. Results in:", CONFIG["output_dir"])


if __name__ == "__main__":
    process_images()
