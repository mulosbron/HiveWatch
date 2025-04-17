# Bee & Wasp Detection

This directory contains scripts for detecting bees and wasps in images using trained YOLO models.

---

## Overview

### 1. Detection Scripts
- `batch_detection.py`: Single model detection for quick processing
- `ensemble_detection.py`: Multi-model ensemble for improved accuracy

### 2. Input/Output Directories
- `source_images/`: Place input images to be processed here
- `detection_results/`: Output from single model detection
- `ensemble_results/`: Output from ensemble detection

---

## Features

### Single Model Detection
- Fast, efficient detection using a single YOLO model
- Lower computational requirements
- Ideal for real-time or resource-limited applications

### Ensemble Detection
- Combines predictions from multiple models for higher accuracy
- Supports different ensemble methods (weighted average, voting, max)
- Uses IoU-based merging to combine overlapping detections
- Automatic performance-based model weighting

---

## Usage

### Single Model Detection
```bash
python batch_detection.py
```

### Ensemble Detection
```bash
python ensemble_detection.py
```

Both scripts will process all images in the `source_images/` directory and save the results to their respective output folders.