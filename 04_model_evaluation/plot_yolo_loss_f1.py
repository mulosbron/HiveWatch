"""
Plot train/val box, class, DFL losses and epoch-wise F1 from a YOLO results.csv,
and save all plots into `<run_folder>_loss_f1` in the current working directory.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    "csv_path": os.path.join(
        CURRENT_DIR, "..", "03_model_training", "runs", "detect",
        "bee_wasp_model_50_320_pre", "results.csv"
    )
}

def ensure_output_dir(csv_path):
    """Create `<run_folder>_loss_f1` in cwd and return its path."""
    run_folder = os.path.basename(os.path.dirname(csv_path))
    out_dir = os.path.join(os.getcwd(), f"{run_folder}_loss_f1")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def plot_and_save(df, x_col, series_info, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for col, label, color in series_info:
        ax.plot(df[x_col], df[col], label=label, color=color)
    ax.set(title=title, xlabel=x_col.capitalize(), ylabel=ylabel)
    ax.legend(frameon=False)
    ax.grid(color="lightgray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def main():
    csv_path = CONFIG["csv_path"]
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    out_dir = ensure_output_dir(csv_path)
    print(f"[INFO] Saving plots to: {out_dir}")

    df = pd.read_csv(csv_path)

    # box loss
    plot_and_save(
        df, "epoch",
        [("train/box_loss", "Train Box Loss", "blue"),
         ("val/box_loss",   "Val Box Loss",   "orange")],
        "Box Loss", "Loss",
        os.path.join(out_dir, "box_loss.png")
    )

    # class loss
    plot_and_save(
        df, "epoch",
        [("train/cls_loss", "Train Class Loss", "blue"),
         ("val/cls_loss",   "Val Class Loss",   "orange")],
        "Class Loss", "Loss",
        os.path.join(out_dir, "class_loss.png")
    )

    # DFL loss
    plot_and_save(
        df, "epoch",
        [("train/dfl_loss", "Train DFL Loss", "blue"),
         ("val/dfl_loss",   "Val DFL Loss",   "orange")],
        "DFL Loss", "Loss",
        os.path.join(out_dir, "dfl_loss.png")
    )

    # F1 score
    p = df["metrics/precision(B)"]
    r = df["metrics/recall(B)"]
    df["F1"] = 2 * (p * r) / (p + r + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["epoch"], df["F1"], label="F1 Score", color="teal",
            linewidth=2, marker="o")
    ax.set(title="Epoch-wise F1 Score", xlabel="Epoch", ylabel="F1 Score")
    ax.set_ylim(0, 1)
    ax.set_xlim(1, df["epoch"].max())
    ax.grid(color="lightgray", linestyle="-", linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_score.png"), dpi=300)
    plt.close(fig)

    print("[INFO] All plots saved (not displayed).")

if __name__ == "__main__":
    main()
