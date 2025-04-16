import os
import yaml
import shutil
import random
import traceback
from tqdm import tqdm
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "data_dir": r"C:\Users\duggy\OneDrive\Belgeler\Github\HiveWatch\03_custom_dataset\bee_vs_wasp_yolo",
    "img_dir": r"C:\Users\duggy\OneDrive\Belgeler\Github\HiveWatch\03_custom_dataset\bee_vs_wasp_yolo\images",
    "label_dir": r"C:\Users\duggy\OneDrive\Belgeler\Github\HiveWatch\03_custom_dataset\bee_vs_wasp_yolo\labels",
    "output_dir": r"C:\Users\duggy\OneDrive\Belgeler\Github\HiveWatch\03_custom_dataset\bee_vs_wasp_yolo\yolo_dataset",
    "image_extensions": ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif', '.ppm'],
    "current_dir": CURRENT_DIR
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def create_dataset_config():
    # Çıktı dizinlerini hazırla
    os.makedirs(os.path.join(CONFIG["output_dir"], "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "test", "labels"), exist_ok=True)

    all_images = [f for f in os.listdir(CONFIG["img_dir"])
                  if os.path.isfile(os.path.join(CONFIG["img_dir"], f)) and
                  os.path.splitext(f)[1].lower() in CONFIG["image_extensions"]]

    num_images = len(all_images)

    # Karıştırma
    random.shuffle(all_images)

    # 70% train, 20% validation, 10% test
    train_size = int(0.7 * num_images)
    val_size = int(0.2 * num_images)

    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]

    print(f"[INFO] Total number of images: {num_images}")
    print(f"[INFO] Training images: {len(train_images)}")
    print(f"[INFO] Validation images: {len(val_images)}")
    print(f"[INFO] Test images: {len(test_images)}")

    def copy_files(image_list, destination):
        for img_file in tqdm(image_list, desc=f"[PROGRESS] Copying files to {destination} directory"):
            try:
                src_img = os.path.join(CONFIG["img_dir"], img_file)
                dst_img = os.path.join(CONFIG["output_dir"], destination, "images", img_file)

                if not os.path.exists(dst_img):
                    shutil.copy(src_img, dst_img)

                label_file = os.path.splitext(img_file)[0] + ".txt"
                src_label = os.path.join(CONFIG["label_dir"], label_file)
                dst_label = os.path.join(CONFIG["output_dir"], destination, "labels", label_file)

                if os.path.exists(src_label) and not os.path.exists(dst_label):  # Label dosyası varsa
                    shutil.copy(src_label, dst_label)
            except Exception as e:
                print(f"[ERROR] Error: Problem copying file {img_file}: {e}")

    if len(os.listdir(os.path.join(CONFIG["output_dir"], "train", "images"))) < len(train_images):
        copy_files(train_images, "train")
    else:
        print("[INFO] Training files already copied, skipping.")

    if len(os.listdir(os.path.join(CONFIG["output_dir"], "val", "images"))) < len(val_images):
        copy_files(val_images, "val")
    else:
        print("[INFO] Validation files already copied, skipping.")

    if len(os.listdir(os.path.join(CONFIG["output_dir"], "test", "images"))) < len(test_images):
        copy_files(test_images, "test")
    else:
        print("[INFO] Test files already copied, skipping.")

    # classes.txt dosyasını okuma
    with open(os.path.join(CONFIG["label_dir"], "classes.txt"), "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # YAML konfigürasyon dosyası oluşturma
    dataset_config = {
        'path': CONFIG["output_dir"],
        'train': os.path.join(CONFIG["output_dir"], "train", "images"),
        'val': os.path.join(CONFIG["output_dir"], "val", "images"),
        'test': os.path.join(CONFIG["output_dir"], "test", "images"),
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes)
    }

    # YAML dosyasına kaydetme
    config_path = os.path.join(CONFIG["output_dir"], "dataset.yaml")
    with open(config_path, "w") as f:
        yaml.dump(dataset_config, f)

    print(f"[INFO] Dataset configuration created: {config_path}")

    return config_path, classes


def train_model():
    # Veri seti konfigürasyonu oluştur
    config_path, classes = create_dataset_config()

    model_folder = os.path.join('runs', 'detect')
    model_name = 'bee_wasp_model_75_640'  # Sabit model adı yerine değişken
    model_path = os.path.join(model_folder, model_name, 'weights', 'last.pt')
    resume_training = os.path.exists(model_path)

    print("[INFO] Starting model training...")
    if resume_training:
        print(f"[INFO] Using previous model (transfer learning): {model_path}")
        model = YOLO(model_path)
        resume_param = True
    else:
        print("[INFO] Creating new model...")
        model = YOLO('yolo11x.pt')
        resume_param = False

    model.train(
        data=config_path,
        epochs=75,
        imgsz=640,
        batch=4,
        workers=6,
        device='cuda',
        name=model_name,
        exist_ok=True,
        resume=resume_param,

        augment=True,  # Rastgele augmentasyon aktif [[1]]
        mosaic=1.0,  # Mosaic augmentasyonu (arka plan gürültüsüne dayanıklılık) [[5]]
        mixup=0.3,  # Mixup ↑ (sınıf sınırlarını netleştirir) [[5]]
        hsv_h=0.02,  # Hue augmentasyonu ↑ (sınıf ayırt etmeyi güçlendirir) [[1]][[7]]
        hsv_s=0.8,  # Doygunluk augmentasyonu ↑ [[1]]
        hsv_v=0.5,  # Parlaklık augmentasyonu ↑ [[1]]
        flipud=0.2,  # Dikey çevirme (sınıf genellemesi) [[1]]
        fliplr=0.5,  # Yatay çevirme (sınıf genellemesi) [[1]]

        scale=0.7,  # Ölçeklendirme aralığı ↑ (kutu ölçek varyasyonu) [[7]]
        translate=0.2,  # Kaydırma oranı ↑ (pozisyon hassasiyeti) [[7]]

        lr0=0.001,  # Başlangıç LR ↑ (daha agresif öğrenme) [[5]]
        lrf=0.01,  # Son LR çarpanı (yavaş öğrenme sonu) [[5]]
        cos_lr=True,  # Kosinüs LR planı (stabil son aşamalar) [[5]]
        warmup_epochs=5,  # Warmup süresi (başlangıç stabilitesi) [[5]]
        warmup_momentum=0.9,  # Momentum ↑ (genelleme için) [[5]]
        warmup_bias_lr=0.1,  # Bias katman LR (başlangıçta hızlı öğrenme) [[5]]

        patience=5  # Erken durdurma sabrı ↑ (overfit önlemek) [[5]]
    )

    # Sınıf isimlerini bir dosyaya kaydet (evaluate_model.py için)
    with open(os.path.join(CONFIG["output_dir"], "classes.txt"), "w") as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    print(f"[INFO] Model training completed. Trained model: runs/train/bee_wasp_model/weights/")


def main():
    print("[INFO] Bee vs WaspHive Object Detection - Model Training")
    print(f"[INFO] Current directory: {CONFIG['current_dir']}")
    print(f"[INFO] Data directory: {CONFIG['data_dir']}")
    print(f"[INFO] Output directory: {CONFIG['output_dir']}")

    try:
        train_model()
        print("[INFO] Model training completed!")
    except Exception as e:
        print(f"[ERROR] Error during program execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()