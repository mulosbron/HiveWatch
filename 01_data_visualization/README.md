# Data Visualization

This directory contains scripts and outputs for visualizing the Bee Dataset. Below is an overview of the visualization process and results.

---

## Purpose
The script `analyze_bee_dataset.py` analyzes the dataset and generates visualizations to better understand the distribution of features (e.g., wasps, varroa mites, pollen) across the dataset. It also provides statistical insights into the dataset's composition.

---

## Key Outputs

### 1. Statistical Summary
The script calculates the following statistics:
- **All three features present:** 0
- **Pairs of features:**
  - Wasp + Varroa: 0
  - Varroa + Pollen: 1
  - Wasp + Pollen: 0
- **Single features:**
  - Wasp: 949
  - Varroa: 1200
  - Pollen: 941
- **No features present:** 3258

### 2. Visualization Categories
The script organizes images into the following categories based on their features:
- `all_three_features`: Images with all three features (wasp, varroa, pollen)
- `wasp_varroa`: Images with both wasp and varroa
- `varroa_pollen`: Images with both varroa and pollen
- `wasp_pollen`: Images with both wasp and pollen
- `only_wasp`: Images with only wasp
- `only_varroa`: Images with only varroa
- `only_pollen`: Images with only pollen
- `none`: Images with none of the features

For each category, random samples are selected and plotted as visualizations.

### 3. Generated Files
- **Statistical Output:** Printed to the console during execution.
- **Visualization Results:** Saved in the `analysis_results` directory as `.png` files, named after each category (e.g., `varroa_pollen.png`, `only_varroa.png`).

---

## How It Works

### 1. Script Workflow
The script performs the following steps:
1. **Data Loading:**  
   - Loads the dataset from `../00_datasets/BeeDataset/data.json`.
   - Filters out entries with the `cooling` property.

2. **Statistical Analysis:**  
   - Calculates the presence of features (single, pairs, or all three) across the dataset.

3. **Image Path Resolution:**  
   - Searches for image files in `images_300`, `images_150`, and `images_200` directories based on resolution priorities.

4. **Visualization Generation:**  
   - Randomly selects images from each category and generates a grid of 2x4 plots.
   - Saves the plots in the `analysis_results` directory.

### 2. Dependencies
The script relies on the following Python libraries:
- `matplotlib.pyplot` (for plotting)

### 3. Execution
To run the script:
```python
python analyze_bee_dataset.py
```