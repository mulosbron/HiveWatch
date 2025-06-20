"""
This script loads BeeDataset/data.json, filters out entries under cooling,
computes counts of images containing wasps, varroa, and pollen in all combinations,
categorizes each image into one of eight feature-based groups,
and for each group saves a grid of sample images to the `analysis_results` folder.
"""

import json
import os
import random
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "dataset_root": os.path.join(CURRENT_DIR, "..", "00_datasets", "BeeDataset"),
    "output_dir": os.path.join(CURRENT_DIR, "analysis_results"),
    "resolutions": ["300", "150", "200"],
    "image_extensions": ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'],
}


def load_data():
    try:
        data_path = os.path.join(CONFIG["dataset_root"], "data.json")
        if not os.path.exists(data_path):
            print(f"[ERROR] Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Successfully loaded data from: {data_path}")
        filtered_data = {k: v for k, v in data.items() if not v['cooling']}
        print(f"[INFO] Filtered out cooling entries: {len(data) - len(filtered_data)} entries removed")
        return filtered_data
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing error in data file: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error loading data: {e}")
        raise


def calculate_stats(data):
    try:
        stats = {
            'all_three': 0,
            'pairs': {
                'wasp_varroa': 0,
                'varroa_pollen': 0,
                'wasp_pollen': 0
            },
            'singles': {
                'wasp': 0,
                'varroa': 0,
                'pollen': 0
            },
            'none': 0
        }

        for props in data.values():
            w = props['wasps']
            v = props['varroa']
            p = props['pollen']

            if w and v and p:
                stats['all_three'] += 1
            elif w and v and not p:
                stats['pairs']['wasp_varroa'] += 1
            elif v and p and not w:
                stats['pairs']['varroa_pollen'] += 1
            elif w and p and not v:
                stats['pairs']['wasp_pollen'] += 1
            elif w and not v and not p:
                stats['singles']['wasp'] += 1
            elif v and not w and not p:
                stats['singles']['varroa'] += 1
            elif p and not w and not v:
                stats['singles']['pollen'] += 1
            else:
                stats['none'] += 1

        print(f"[INFO] Statistics calculated for {len(data)} entries")
        return stats
    except KeyError as e:
        print(f"[ERROR] Missing key in data structure: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Error calculating statistics: {e}")
        raise


def find_image_path(img_name):
    try:
        for res in CONFIG["resolutions"]:
            img_dir = os.path.join(CONFIG["dataset_root"], f"images_{res}")
            path = os.path.join(img_dir, img_name)
            if os.path.exists(path):
                return path

        print(f"[WARNING] Image not found in any resolution directory: {img_name}")
        return None
    except Exception as e:
        print(f"[ERROR] Error finding image path for {img_name}: {e}")
        return None


def plot_samples(category_name, image_names, rows=2, cols=4):
    try:
        print(f"[INFO] Creating plot for category: {category_name}")
        plt.figure(figsize=(20, 10))
        plt.suptitle(category_name, fontsize=16)

        found_images = 0
        for i, img_name in enumerate(image_names[:8], 1):
            img_path = find_image_path(img_name)
            if not img_path:
                continue

            try:
                img = plt.imread(img_path)
                plt.subplot(rows, cols, i)
                plt.imshow(img)
                plt.axis('off')
                found_images += 1
            except Exception as e:
                print(f"[ERROR] Could not read or display image {img_name}: {e}")

        plt.tight_layout()
        output_file = os.path.join(CONFIG["output_dir"], f"{category_name}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"[INFO] Plot saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Error creating plot for {category_name}: {e}")


def main():
    try:
        print("[INFO] Starting analysis process")

        try:
            os.makedirs(CONFIG["output_dir"], exist_ok=True)
            print(f"[INFO] Output directory ensured: {CONFIG['output_dir']}")
        except Exception as e:
            print(f"[ERROR] Could not create output directory: {e}")
            return

        try:
            data = load_data()
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return

        try:
            stats = calculate_stats(data)
        except Exception as e:
            print(f"[ERROR] Failed to calculate statistics: {e}")
            return

        print(f"[INFO] All three features: {stats['all_three']}")
        print("[INFO] Pairwise Combinations:")
        for k, v in stats['pairs'].items():
            print(f"[INFO] - {k.replace('_', ' + ')}: {v}")
        print("[INFO] Single Features:")
        for k, v in stats['singles'].items():
            print(f"[INFO] - {k}: {v}")
        print(f"[INFO] No features: {stats['none']}")

        categories = {
            'all_three_features': [],
            'wasp_varroa': [],
            'varroa_pollen': [],
            'wasp_pollen': [],
            'only_wasp': [],
            'only_varroa': [],
            'only_pollen': [],
            'none': []
        }

        print("[INFO] Categorizing images")
        for img_name, props in data.items():
            w = props['wasps']
            v = props['varroa']
            p = props['pollen']

            if w and v and p:
                categories['all_three_features'].append(img_name)
            elif w and v:
                categories['wasp_varroa'].append(img_name)
            elif v and p:
                categories['varroa_pollen'].append(img_name)
            elif w and p:
                categories['wasp_pollen'].append(img_name)
            elif w:
                categories['only_wasp'].append(img_name)
            elif v:
                categories['only_varroa'].append(img_name)
            elif p:
                categories['only_pollen'].append(img_name)
            else:
                categories['none'].append(img_name)

        print("[INFO] Creating sample plots for each category")
        for cat_name, images in categories.items():
            if images:
                random.shuffle(images)
                plot_samples(cat_name, images)
            else:
                print(f"[WARNING] No images found for category: {cat_name}")

        print("[INFO] Analysis completed successfully")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()