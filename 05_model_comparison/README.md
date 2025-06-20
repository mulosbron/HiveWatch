# Model Comparison

This directory contains scripts and outputs for comparing multiple trained YOLO models on the Bee vs Wasp dataset.

---

## Overview

### 1. `model_comparison.py`
The primary script that evaluates and compares multiple YOLO models using the same test images.

### 2. `model_comparison_results/`
Contains all output visualizations and the comprehensive comparison report.

---

## Purpose

The comparison script analyzes and compares models by:
1. Evaluating detection and classification performance across multiple models
2. Computing precision, recall, F1-score, and IoU metrics for each model
3. Measuring inference speed on the same set of test images
4. Generating comparative visualizations
5. Creating a detailed summary report

---

## Key Outputs

### 1. Visualizations
- Performance metrics: `metrics_comparison.png`
- Confusion matrices: `confusion_matrices.png`
- Detection counts: `detection_counts.png`
- Inference times: `inference_times.png`
- Confidence distributions: `confidence_distribution.png`
- Class prediction distributions: `class_distribution.png`

### 2. Reports
- `model_comparison_report.txt`: Comprehensive comparison with metrics tables
- `debug_info.txt`: Detailed debug information (when run with debug enabled)

### 3. Execution
To run the comparison:
```bash
python model_comparison.py
```