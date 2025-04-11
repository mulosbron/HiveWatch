import json
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt

# ================ #
# SETTINGS         #
# ================ #
DATASET_ROOT = Path(__file__).parent.parent / "00_datasets" / "BeeDataset"
RESOLUTIONS = ["300", "150", "200"]
OUTPUT_DIR = Path("analysis_results")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================ #
# DATA LOADING     #
# ================ #
def load_data():
    data_path = DATASET_ROOT / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not v['cooling']}  # Filter out 'cooling'


# ================ #
# STATISTICS       #
# ================ #
def calculate_stats(data):
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

        # All three
        if w and v and p:
            stats['all_three'] += 1
        # Pairs
        elif w and v and not p:
            stats['pairs']['wasp_varroa'] += 1
        elif v and p and not w:
            stats['pairs']['varroa_pollen'] += 1
        elif w and p and not v:
            stats['pairs']['wasp_pollen'] += 1
        # Singles
        elif w and not v and not p:
            stats['singles']['wasp'] += 1
        elif v and not w and not p:
            stats['singles']['varroa'] += 1
        elif p and not w and not v:
            stats['singles']['pollen'] += 1
        # None
        else:
            stats['none'] += 1

    return stats


# ================ #
# VISUALIZATION    #
# ================ #
def find_image_path(img_name):
    for res in RESOLUTIONS:
        path = DATASET_ROOT / f"images_{res}" / img_name
        if path.exists():
            return path
    return None


def plot_samples(category_name, image_names, rows=2, cols=4):
    plt.figure(figsize=(20, 10))
    plt.suptitle(category_name, fontsize=16)

    for i, img_name in enumerate(image_names[:8], 1):
        img_path = find_image_path(img_name)
        if not img_path:
            continue

        img = plt.imread(img_path)
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{category_name}.png")
    plt.close()


# ================ #
# MAIN PROCESS     #
# ================ #
def main():
    data = load_data()
    stats = calculate_stats(data)

    # Show statistics
    print("=== Basic Statistics ===")
    print(f"All three features: {stats['all_three']}")
    print("\nPairwise Combinations:")
    for k, v in stats['pairs'].items():
        print(f"- {k.replace('_', ' + ')}: {v}")
    print("\nSingle Features:")
    for k, v in stats['singles'].items():
        print(f"- {k}: {v}")
    print(f"\nNo features: {stats['none']}")

    # Collect visual examples
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

    # Select random samples and plot
    for cat_name, images in categories.items():
        if images:
            random.shuffle(images)
            plot_samples(cat_name, images)


if __name__ == "__main__":
    main()
