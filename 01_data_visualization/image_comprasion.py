import os
from PIL import Image
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "ensemble_dir": r"C:\Users\duggy\OneDrive\Belgeler\Github\HiveWatch\08_image_tracking\ensemble_results1",
    "output_dir": r"C:\Users\duggy\OneDrive\Belgeler\Github\HiveWatch\08_image_tracking\output",
    "comparison_dir": os.path.join(CURRENT_DIR, "comparison_output"),
    "valid_extensions": ['.jpg', '.jpeg', '.png'],
    "output_format": "jpg",
    "jpg_quality": 95,
    "dpi": 300
}


def get_image_files(directory):
    try:
        image_files = []

        if not os.path.exists(directory):
            print(f"[ERROR] Directory does not exist: {directory}")
            return image_files

        for filename in os.listdir(directory):
            try:
                file_ext = os.path.splitext(filename.lower())[1]
                if file_ext in CONFIG["valid_extensions"]:
                    image_files.append(os.path.join(directory, filename))
            except Exception as e:
                print(f"[ERROR] Error processing file {filename}: {e}")

        return image_files
    except Exception as e:
        print(f"[ERROR] Error getting image files from {directory}: {e}")
        return []


def find_matching_files(ensemble_files, output_files):
    try:
        matches = []

        ensemble_dict = {os.path.basename(f): f for f in ensemble_files}
        output_dict = {os.path.basename(f): f for f in output_files}

        for filename in ensemble_dict:
            if filename in output_dict:
                matches.append((ensemble_dict[filename], output_dict[filename]))

        return matches
    except Exception as e:
        print(f"[ERROR] Error finding matching files: {e}")
        return []


def compare_and_save_images(matches):
    count = 0

    for ensemble_path, output_path in matches:
        try:
            ensemble_pil = Image.open(ensemble_path)
            output_pil = Image.open(output_path)

            width1, height1 = ensemble_pil.size
            width2, height2 = output_pil.size

            max_height = max(height1, height2)

            fig_width = (width1 + width2) / CONFIG["dpi"]
            fig_height = (max_height / CONFIG["dpi"]) + 1

            fig = Figure(figsize=(fig_width, fig_height), dpi=CONFIG["dpi"])
            canvas = FigureCanvas(fig)

            grid = fig.add_gridspec(2, 2, height_ratios=[0.1, 1], hspace=0.3)

            title_ax = fig.add_subplot(grid[0, :])
            title_ax.axis('off')
            filename = os.path.basename(ensemble_path)
            title_ax.text(0.5, 0.5, f"Comparison: {filename}",
                          fontsize=16, ha='center', va='center')

            ax1 = fig.add_subplot(grid[1, 0])
            ax2 = fig.add_subplot(grid[1, 1])

            ax1.imshow(np.array(ensemble_pil))
            ax1.set_title('Ensemble Result')
            ax1.axis('off')

            ax2.imshow(np.array(output_pil))
            ax2.set_title('Output')
            ax2.axis('off')

            base_filename = os.path.splitext(filename)[0]
            output_ext = f".{CONFIG['output_format']}"
            save_path = os.path.join(CONFIG["comparison_dir"], f"compare_{base_filename}{output_ext}")

            if CONFIG["output_format"].lower() == "jpg":
                try:
                    canvas.draw()
                    buf = canvas.buffer_rgba()
                    img = Image.frombuffer('RGBA', canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
                    img = img.convert('RGB')
                    img.save(save_path, format="JPEG", quality=CONFIG["jpg_quality"])
                except Exception as e:
                    print(f"[ERROR] Error saving JPEG image {save_path}: {e}")
                    raise
            else:
                try:
                    fig.savefig(save_path, format="png", dpi=CONFIG["dpi"], bbox_inches="tight")
                except Exception as e:
                    print(f"[ERROR] Error saving PNG image {save_path}: {e}")
                    raise

            count += 1
            print(f"[PROGRESS] ({count}/{len(matches)}) Saved comparison for: {filename}")

        except Exception as e:
            print(f"[ERROR] Processing {os.path.basename(ensemble_path)}: {e}")

    return count


def main():
    try:
        print(f"[STATUS] Starting image comparison process")

        try:
            os.makedirs(CONFIG["comparison_dir"], exist_ok=True)
            print(f"[INFO] Output directory ensured: {CONFIG['comparison_dir']}")
        except Exception as e:
            print(f"[ERROR] Failed to create comparison directory: {e}")
            return

        print("[STATUS] Getting image files from directories...")
        ensemble_files = get_image_files(CONFIG["ensemble_dir"])
        if not ensemble_files:
            print(f"[WARNING] No images found in ensemble directory: {CONFIG['ensemble_dir']}")

        output_files = get_image_files(CONFIG["output_dir"])
        if not output_files:
            print(f"[WARNING] No images found in output directory: {CONFIG['output_dir']}")

        print(f"[INFO] Found {len(ensemble_files)} images in ensemble directory")
        print(f"[INFO] Found {len(output_files)} images in output directory")

        matches = find_matching_files(ensemble_files, output_files)
        print(f"[INFO] Found {len(matches)} matching image files")

        if not matches:
            print("[WARNING] No matching image files found. Nothing to compare.")
            return

        print(f"[STATUS] Creating high-resolution comparison images in {CONFIG['output_format'].upper()} format...")
        count = compare_and_save_images(matches)

        print(f"[STATUS] Comparison complete")
        print(f"[INFO] Created {count} comparison images")
        print(f"[INFO] Output format: {CONFIG['output_format'].upper()} at {CONFIG['dpi']} DPI")
        if CONFIG["output_format"].lower() == "jpg":
            print(f"[INFO] JPEG quality: {CONFIG['jpg_quality']}")
        print(f"[INFO] Comparison images saved to: {os.path.abspath(CONFIG['comparison_dir'])}")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

    print("[STATUS] Process finished")


if __name__ == "__main__":
    main()