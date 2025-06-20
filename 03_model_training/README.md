# Model Training

This directory contains scripts for training YOLO models on bee-related datasets (Bee vs Wasp and Pollen vs Varroa).

---

## Key Features
- **YOLOv11x** base model
- Automated dataset preparation and splitting (70% train, 20% validation, 10% test)
- Extensive data augmentation (mosaic, mixup, HSV, flipping)
- GPU acceleration with CUDA
- Transfer learning support (resumes training if a previous model exists)

---

## Available Training Scripts
- `bee_wasp_prepare_and_train.py` - For Bee vs Wasp detection
- `pollen_varroa_prepare_and_train.py` - For Pollen vs Varroa detection

---

## Execution
```bash
# Install dependencies
pip install ultralytics tqdm pyyaml
```

```bash
# Run Bee vs Wasp training
python bee_wasp_prepare_and_train.py
```

```bash
# Run Pollen vs Varroa training
python pollen_varroa_prepare_and_train.py
```