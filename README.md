# HiveWatch

An AI-powered monitoring and detection system for identifying bees, wasps, and potential threats to beehives.

---

## Project Overview

HiveWatch uses computer vision and machine learning to help beekeepers monitor hive activity, detect threats, and maintain healthy colonies. The system can identify and differentiate between bees, wasps, and other insects that may pose risks to the hive.

---

## Requirements

- Python 3.9.20
- Dependencies listed in requirements.txt

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Key Components

### 1. Dataset Management
- Custom datasets for bee and wasp detection
- Background image processing for negative samples
- Data visualization tools for dataset analysis

### 2. Model Development
- YOLO-based object detection models
- Multiple model configurations for different performance needs
- Background generation for synthetic data augmentation

### 3. Analysis & Evaluation
- Model metrics and performance evaluation
- Comprehensive visualizations of detection accuracy
- Comparative analysis between model variants

### 4. Detection Tools
- Single-model detector for efficient processing
- Ensemble detection for higher accuracy
- Batch processing capabilities for multiple images

---

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place images to be analyzed in the `source_images/` directory

3. Run detection using either:
```bash
python batch_detection.py  # Single model detection
```
or
```bash
python ensemble_detection.py  # Ensemble detection
```

4. View detection results in the corresponding output directory

---

## Project Structure

- `00_datasets/`: Source datasets and preprocessing scripts
- `01_data_visualization/`: Tools for dataset analysis and visualization
- `02_background_generation/`: Scripts for generating synthetic backgrounds
- `03_custom_dataset/`: Prepared datasets in YOLO format
- `04_model_training/`: Model training scripts and saved models
- `05_model_metrics/`: Model evaluation and analysis
- `07_image_tracking/`: Detection scripts and visualization tools

---

## Future Development

- Real-time video processing
- Mobile application for field use
- Integration with automated hive monitoring systems
- Expanded detection capabilities for additional threats