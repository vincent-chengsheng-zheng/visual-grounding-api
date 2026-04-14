"""
scripts/accuracy_at_threshold.py

Compute accuracy@threshold (% predictions with IoU > threshold).
Standard metric for visual grounding tasks.

Output:
    results/accuracy_at_threshold.json
    results/accuracy_at_threshold.png

Usage:
    python scripts/accuracy_at_threshold.py
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def compute_iou(pred, gt):
    p_x1 = pred[0] - pred[2] / 2
    p_y1 = pred[1] - pred[3] / 2
    p_x2 = pred[0] + pred[2] / 2
    p_y2 = pred[1] + pred[3] / 2
    g_x1 = gt[0] - gt[2] / 2
    g_y1 = gt[1] - gt[3] / 2
    g_x2 = gt[0] + gt[2] / 2
    g_y2 = gt[1] + gt[3] / 2
    ix1, iy1 = max(p_x1, g_x1), max(p_y1, g_y1)
    ix2, iy2 = min(p_x2, g_x2), min(p_y2, g_y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = pred[2] * pred[3] + gt[2] * gt[3] - inter
    return inter / max(union, 1e-6)


def main():
    thresholds = [0.1, 0.25, 0.5, 0.75]
    results = {}

    for model_name in ["main", "ablation"]:
        pred_path = os.path.join(RESULTS_DIR, model_name, "predictions.json")
        if not os.path.exists(pred_path):
            continue
        with open(pred_path) as f:
            predictions = json.load(f)

        ious = [compute_iou(p["pred_bbox"], p["gt_bbox"]) for p in predictions]
        n = len(ious)

        model_results = {"n_samples": n, "mean_iou": round(float(np.mean(ious)), 4)}
        for t in thresholds:
            acc = sum(1 for iou in ious if iou >= t) / n
            model_results[f"acc@{t}"] = round(acc * 100, 1)

        results[model_name] = model_results

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "accuracy_at_threshold.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    # Print
    print(f"\n{'='*60}")
    print(f"  ACCURACY @ IoU THRESHOLD")
    print(f"{'='*60}")
    print(f"  {'Model':<12s} {'Mean IoU':>9s}", end="")
    for t in thresholds:
        print(f"  {'@'+str(t):>6s}", end="")
    print()
    for model, data in results.items():
        print(f"  {model:<12s} {data['mean_iou']:>9.4f}", end="")
        for t in thresholds:
            print(f"  {data[f'acc@{t}']:>5.1f}%", end="")
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(thresholds))
    bar_width = 0.35
    colors = {"main": "#4CAF50", "ablation": "#FF9800"}

    for i, (model, data) in enumerate(results.items()):
        accs = [data[f"acc@{t}"] for t in thresholds]
        bars = ax.bar(x + i * bar_width, accs, bar_width, label=model,
                      color=colors.get(model, "#999"), edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels([f"IoU > {t}" for t in thresholds])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy @ IoU Threshold (Val Set, 1000 samples)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "accuracy_at_threshold.png")
    plt.savefig(out_png, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
