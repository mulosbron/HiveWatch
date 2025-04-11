# Background Images

This directory contains scripts and outputs for managing background images used in the project. The primary script selects random images from a source directory, copies them to a target folder, and generates corresponding empty label files (`.txt`) for each image.

---

## Purpose
The script is designed to:
1. Select a specified number of random background images from a source directory.
2. Copy these images to the `background_images/imgs` directory.
3. Create empty `.txt` label files in the `background_images/labels` directory for each image.

These background images and labels are typically used in object detection tasks where negative samples (images without objects of interest) are required.

---

## Key Outputs

### 1. Directories Created
- **Images Directory:** `background_images/imgs`  
  Contains the randomly selected background images.
- **Labels Directory:** `background_images/labels`  
  Contains empty `.txt` files corresponding to each image.

### 2. Execution
To run the script:
```python
python prepare_background_images.py
```