import os
import cv2
import numpy as np
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "data_dir": os.path.join(CURRENT_DIR, "..", "03_custom_dataset", "pollen_vs_varroa_yolo"),
    "source_img_dir": None,
    "source_label_dir": None,
    "output_dir": None,
    "output_img_dir": None,
    "output_label_dir": None,
    "image_extensions": ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif', '.ppm'],
    "target_size": 320,
    "num_workers": 8,
    "gaussian_blur_kernel": (3, 3),
    "gaussian_blur_sigma": 1.0,
}

# set derived paths
CONFIG["source_img_dir"]   = os.path.join(CONFIG["data_dir"], "images")
CONFIG["source_label_dir"] = os.path.join(CONFIG["data_dir"], "labels")
CONFIG["output_dir"]       = os.path.join(CONFIG["data_dir"], "preprocessed_data")
CONFIG["output_img_dir"]   = os.path.join(CONFIG["output_dir"], "images")
CONFIG["output_label_dir"] = os.path.join(CONFIG["output_dir"], "labels")


def create_directories():
    os.makedirs(CONFIG["output_img_dir"], exist_ok=True)
    os.makedirs(CONFIG["output_label_dir"], exist_ok=True)
    print(f"[INFO] Created output directories under {CONFIG['output_dir']}")


def apply_image_processing(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur and normalize to [0,1] float."""
    blurred = cv2.GaussianBlur(
        image,
        CONFIG["gaussian_blur_kernel"],
        CONFIG["gaussian_blur_sigma"]
    )
    return blurred.astype(np.float32) / 255.0


def letterbox_resize(image: np.ndarray, target_size: int):
    """
    Resize image with aspect ratio preserved using letterbox (padding).
    Returns padded square image, scale, x_offset, y_offset, orig_w, orig_h.
    """
    orig_h, orig_w = image.shape[:2]
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize((image * 255).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_AREA)
    # create gray canvas
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    x_off = (target_size - new_w) // 2
    y_off = (target_size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas, scale, x_off, y_off, orig_w, orig_h


def adjust_yolo_labels(label_path, scale, x_off, y_off, orig_w, orig_h, target_size):
    """
    Read original YOLO labels, convert to absolute coords, apply letterbox scale+offset,
    then re-normalize to [0,1] over target_size.
    """
    if not os.path.exists(label_path):
        return []
    adjusted = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = parts[0]
            x_c, y_c, w, h = map(float, parts[1:])
            # absolute on orig
            x_abs = x_c * orig_w
            y_abs = y_c * orig_h
            w_abs = w * orig_w
            h_abs = h * orig_h
            # scale & offset
            x_scaled = x_abs * scale + x_off
            y_scaled = y_abs * scale + y_off
            w_scaled = w_abs * scale
            h_scaled = h_abs * scale
            # normalize to target_size
            x_cn = x_scaled / target_size
            y_cn = y_scaled / target_size
            w_n = w_scaled / target_size
            h_n = h_scaled / target_size
            # clip
            x_cn = max(0.0, min(1.0, x_cn))
            y_cn = max(0.0, min(1.0, y_cn))
            w_n  = max(0.0, min(1.0, w_n))
            h_n  = max(0.0, min(1.0, h_n))
            adjusted.append(f"{cls_id} {x_cn:.6f} {y_cn:.6f} {w_n:.6f} {h_n:.6f}")
    return adjusted


def process_image(image_path):
    """Process one image: blur, letterbox resize, save, adjust labels and save."""
    try:
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)
        lbl_path = os.path.join(CONFIG["source_label_dir"], f"{name}.txt")
        # detect if background (empty or missing label)
        is_bg = True
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                if f.read().strip():
                    is_bg = False
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARNING] Cannot read {filename}")
            return None
        # blur + normalize
        proc = apply_image_processing(img)
        # letterbox resize
        canvas, scale, x_off, y_off, orig_w, orig_h = letterbox_resize(proc, CONFIG["target_size"])
        # save image
        out_img_path = os.path.join(CONFIG["output_img_dir"], filename)
        cv2.imwrite(out_img_path, canvas)
        # save label
        out_lbl_path = os.path.join(CONFIG["output_label_dir"], f"{name}.txt")
        if is_bg:
            open(out_lbl_path, 'w').close()
        else:
            labels = adjust_yolo_labels(lbl_path, scale, x_off, y_off, orig_w, orig_h, CONFIG["target_size"])
            if labels:
                with open(out_lbl_path, 'w') as f:
                    f.write("\n".join(labels))
            else:
                open(out_lbl_path, 'w').close()
        return out_img_path
    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        traceback.print_exc()
        return None


def preprocess_dataset():
    create_directories()
    # gather images
    imgs = []
    for ext in CONFIG["image_extensions"]:
        imgs += [os.path.join(CONFIG["source_img_dir"], f)
                 for f in os.listdir(CONFIG["source_img_dir"]) if f.lower().endswith(ext)]
    total = len(imgs)
    print(f"[INFO] {total} images to process (letterbox -> {CONFIG['target_size']}x{CONFIG['target_size']})")
    # process in batches
    batch = 100
    done = 0
    for i in range(0, total, batch):
        chunk = imgs[i:i+batch]
        print(f"[INFO] Batch {i//batch+1}/{(total+batch-1)//batch}")
        with ProcessPoolExecutor(max_workers=CONFIG["num_workers"]) as exe:
            list(tqdm(exe.map(process_image, chunk), total=len(chunk)))
        done += len(chunk)
        print(f"[INFO] {done}/{total} done")
    # write config yaml
    cfg = {
        "preprocessing": {
            "source_dir": CONFIG["data_dir"],
            "output_dir": CONFIG["output_dir"],
            "target_size": CONFIG["target_size"],
            "letterbox": True,
            "gaussian_blur": {
                "kernel": CONFIG["gaussian_blur_kernel"],
                "sigma": CONFIG["gaussian_blur_sigma"]
            }
        }
    }
    with open(os.path.join(CONFIG["output_dir"], "preprocessing_config.yaml"), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[INFO] Preprocessing complete. Results in {CONFIG['output_dir']}")


def main():
    print("[INFO] Starting preprocessing with letterbox (aspect ratio preserved)")
    try:
        preprocess_dataset()
        print("[INFO] All done!")
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
