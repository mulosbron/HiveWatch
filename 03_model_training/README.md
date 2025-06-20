# Model Training

This directory contains scripts for training YOLO models on bee-related datasets (Bee vs Wasp and Pollen vs Varroa).

---

## Key Features
- **YOLOv11x** base model
- Automated dataset preparation and splitting (70% train, 20% validation, 10% test)
- Extensive data augmentation (mosaic, mixup, HSV, flipping)
- GPU acceleration with CUDA
- Transfer learning support (resumes training if a previous model exists)
