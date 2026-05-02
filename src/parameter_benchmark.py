"""
parameter_benchmark.py
----------------------
Parameter sensitivity study for the Haar cascade face detector.

Sweeps two key detector parameters across a grid:
  - scaleFactor    : 1.05 to 1.40 (controls image pyramid steps)
  - minNeighbors   : 2 to 8       (controls detection confidence threshold)

For each parameter combination, runs detection on all sample images
and records:
  - Total faces detected
  - Detection rate (% of images with at least 1 face)
  - Mean processing time

Generates:
  1. Heatmap — total detections across parameter grid
  2. Heatmap — detection rate across parameter grid
  3. Heatmap — mean processing time across parameter grid
  4. Line plot — faces detected vs minNeighbors at fixed scaleFactor
  5. Precision/recall trade-off — sensitivity vs specificity

Engineering insight: Lower scaleFactor + lower minNeighbors → more
detections (higher recall, more false positives). Higher values →
fewer detections (higher precision, more misses). This study identifies
the optimal operating point for the sample image set.

Inputs : data/sample_images/ (processed directly — no JSON required)
Outputs: results/param_benchmark_heatmap.png
         results/param_benchmark_sensitivity.png
         results/param_benchmark_tradeoff.png

Usage:
    python src/parameter_benchmark.py
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))
from face_detector import FaceDetector

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT_DIR, "data", "sample_images")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# ── Parameter grid ────────────────────────────────────────────────────────────
SCALE_FACTORS   = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]
MIN_NEIGHBOURS  = [2, 3, 4, 5, 6, 7, 8]

# Ground truth face counts (from image_metadata.csv)
GROUND_TRUTH = {
    "test1_single_face.jpg":  1,
    "test2_two_faces.jpg":    2,
    "test3_gradient_bg.jpg":  1,
    "test4_group.jpg":        3,
    "test5_small_face.jpg":   1,
    "test6_dark_bg.jpg":      1,
}

SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}


# ── Benchmark runner ──────────────────────────────────────────────────────────

def load_images(data_dir):
    """Load all sample images into memory once."""
    images = {}
    for fname in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXT:
            continue
        path = os.path.join(data_dir, fname)
        img  = cv2.imread(path)
        if img is not None:
            images[fname] = img
    return images


def run_benchmark(images, scale_factors, min_neighbours):
    """
    Run detection across full parameter grid.
    Returns dict of results indexed by (scale, neighbours).
    """
    n_sf  = len(scale_factors)
    n_mn  = len(min_neighbours)

    total_detections  = np.zeros((n_sf, n_mn))
    detection_rates   = np.zeros((n_sf, n_mn))
    mean_times        = np.zeros((n_sf, n_mn))
    true_positives    = np.zeros((n_sf, n_mn))
    false_positives   = np.zeros((n_sf, n_mn))

    total_combos = n_sf * n_mn
    done = 0

    for i, sf in enumerate(scale_factors):
        for j, mn in enumerate(min_neighbours):
            detector = FaceDetector(face_scale=sf, face_min_neighbours=mn)

            counts = []
            times  = []
            tp = fp = 0

            for fname, image in images.items():
                result = detector.detect(image)
                counts.append(result["face_count"])
                times.append(result["processing_ms"])

                # Compare to ground truth
                gt = GROUND_TRUTH.get(fname, None)
                if gt is not None:
                    detected = result["face_count"]
                    tp += min(detected, gt)           # correctly found faces
                    fp += max(0, detected - gt)       # extra false detections

            total_detections[i, j] = sum(counts)
            detection_rates[i, j]  = sum(1 for c in counts if c > 0) / len(counts)
            mean_times[i, j]       = np.mean(times)
            true_positives[i, j]   = tp
            false_positives[i, j]  = fp

            done += 1
            print(f"  [{done:>2}/{total_combos}] "
                  f"scale={sf:.2f} neighbours={mn}  "
                  f"→ {sum(counts)} faces  "
                  f"{mean_times[i,j]:.0f}ms")

    return {
        "total_detections": total_detections,
        "detection_rates":  detection_rates,
        "mean_times":       mean_times,
        "true_positives":   true_positives,
        "false_positives":  false_positives,
    }


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_heatmaps(data, scale_factors, min_neighbours, output_path):
    """Three heatmaps: detections, detection rate, processing time."""
    sf_labels = [f"{s:.2f}" for s in scale_factors]
    mn_labels  = [str(m) for m in min_neighbours]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Haar Cascade Parameter Sensitivity Study\n"
        "scaleFactor vs minNeighbors — 6 sample images",
        fontsize=12, fontweight="bold"
    )

    panels = [
        (data["total_detections"], "Total Faces Detected",
         "Greens", "%.0f"),
        (data["detection_rates"] * 100, "Detection Rate (%)",
         "Blues", "%.0f%%"),
        (data["mean_times"], "Mean Processing Time (ms)",
         "Oranges", "%.0f"),
    ]

    for ax, (matrix, title, cmap, fmt) in zip(axes, panels):
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85)

        for i in range(len(scale_factors)):
            for j in range(len(min_neighbours)):
                val = matrix[i, j]
                ax.text(j, i, fmt % val,
                        ha="center", va="center",
                        fontsize=8.5, fontweight="bold",
                        color="white" if val > matrix.max()*0.65 else "black")

        ax.set_xticks(range(len(min_neighbours)))
        ax.set_yticks(range(len(scale_factors)))
        ax.set_xticklabels(mn_labels, fontsize=9)
        ax.set_yticklabels(sf_labels, fontsize=9)
        ax.set_xlabel("minNeighbors", fontsize=10)
        ax.set_ylabel("scaleFactor", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

    # Mark optimal operating point (highest true positive, lowest false positive)
    tp = data["true_positives"]
    fp = data["false_positives"]
    score = tp - fp * 0.5
    best_i, best_j = np.unravel_index(score.argmax(), score.shape)
    for ax in axes:
        ax.add_patch(plt.Rectangle(
            (best_j-0.5, best_i-0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=2.5,
            label=f"Optimal: sf={scale_factors[best_i]:.2f}, "
                  f"mn={min_neighbours[best_j]}"
        ))
    axes[0].legend(fontsize=8, loc="upper right",
                   framealpha=0.9, edgecolor="red")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_sensitivity(data, scale_factors, min_neighbours, output_path):
    """Line plot: detections vs minNeighbors at three scaleFactor levels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Parameter Sensitivity — Effect on Detection Count\n"
        "Higher minNeighbors = stricter filtering (fewer false positives)",
        fontsize=12, fontweight="bold"
    )

    selected_sf = [0, len(scale_factors)//2, -1]
    colours     = ["#D85A30", "#1A6BAD", "#1D9E75"]

    for idx, colour in zip(selected_sf, colours):
        sf    = scale_factors[idx]
        dets  = data["total_detections"][idx, :]
        times = data["mean_times"][idx, :]

        axes[0].plot(min_neighbours, dets, "o-", color=colour,
                     linewidth=2.0, markersize=6,
                     label=f"scaleFactor={sf:.2f}")
        axes[1].plot(min_neighbours, times, "o-", color=colour,
                     linewidth=2.0, markersize=6,
                     label=f"scaleFactor={sf:.2f}")

    for ax, ylabel, title in zip(
        axes,
        ["Total Faces Detected", "Mean Processing Time (ms)"],
        ["Detection Count vs minNeighbors",
         "Processing Time vs minNeighbors"]
    ):
        ax.set_xlabel("minNeighbors", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#FAFAFA")
        ax.set_xticks(min_neighbours)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_tradeoff(data, scale_factors, min_neighbours, output_path):
    """Precision/recall trade-off across parameter grid."""
    gt_total = sum(GROUND_TRUTH.values())

    tp_flat = data["true_positives"].flatten()
    fp_flat = data["false_positives"].flatten()

    precision = np.where(tp_flat+fp_flat > 0,
                         tp_flat / (tp_flat+fp_flat), 0)
    recall    = tp_flat / gt_total

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(recall, precision,
                    c=range(len(recall)), cmap="viridis",
                    s=80, zorder=3, edgecolors="white", linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="Parameter combination index")

    # Label optimal point
    f1 = np.where(precision+recall > 0,
                  2*precision*recall/(precision+recall), 0)
    best = f1.argmax()
    ax.annotate(
        f"Best F1={f1[best]:.2f}\n"
        f"P={precision[best]:.2f} R={recall[best]:.2f}",
        xy=(recall[best], precision[best]),
        xytext=(recall[best]-0.15, precision[best]-0.12),
        fontsize=9, color="#D85A30",
        arrowprops=dict(arrowstyle="->", color="#D85A30", lw=1.2)
    )
    ax.scatter(recall[best], precision[best],
               s=200, color="#D85A30", zorder=5, marker="*")

    ax.set_xlabel("Recall  (detected faces / total expected faces)", fontsize=11)
    ax.set_ylabel("Precision  (correct detections / total detections)", fontsize=11)
    ax.set_title(
        "Precision-Recall Trade-off — Parameter Grid\n"
        "Each point = one (scaleFactor, minNeighbors) combination",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Parameter Benchmark — Haar Cascade Sensitivity Study")
    print(f"  scaleFactor range  : {SCALE_FACTORS[0]} → {SCALE_FACTORS[-1]} "
          f"({len(SCALE_FACTORS)} values)")
    print(f"  minNeighbors range : {MIN_NEIGHBOURS[0]} → {MIN_NEIGHBOURS[-1]} "
          f"({len(MIN_NEIGHBOURS)} values)")
    print(f"  Total combinations : {len(SCALE_FACTORS)*len(MIN_NEIGHBOURS)}")
    print(f"  Images per combo   : {len(GROUND_TRUTH)}")
    print(f"\nRunning benchmark ({len(SCALE_FACTORS)*len(MIN_NEIGHBOURS)} "
          f"parameter combinations)...\n")

    images = load_images(DATA_DIR)
    if not images:
        print(f"  No images found in {DATA_DIR}")
        return

    data = run_benchmark(images, SCALE_FACTORS, MIN_NEIGHBOURS)

    # Find optimal
    tp = data["true_positives"]
    fp = data["false_positives"]
    score = tp - fp * 0.5
    bi, bj = np.unravel_index(score.argmax(), score.shape)

    print(f"\n  Optimal parameters:")
    print(f"    scaleFactor  : {SCALE_FACTORS[bi]:.2f}")
    print(f"    minNeighbors : {MIN_NEIGHBOURS[bj]}")
    print(f"    True positives  : {int(tp[bi,bj])}")
    print(f"    False positives : {int(fp[bi,bj])}")

    print("\nGenerating plots...")
    plot_heatmaps(data, SCALE_FACTORS, MIN_NEIGHBOURS,
                  os.path.join(RESULTS_DIR, "param_benchmark_heatmap.png"))
    plot_sensitivity(data, SCALE_FACTORS, MIN_NEIGHBOURS,
                     os.path.join(RESULTS_DIR, "param_benchmark_sensitivity.png"))
    plot_tradeoff(data, SCALE_FACTORS, MIN_NEIGHBOURS,
                  os.path.join(RESULTS_DIR, "param_benchmark_tradeoff.png"))

    print("\nParameter benchmark complete. 3 plots saved to results/")


if __name__ == "__main__":
    main()
