"""
scripts/bias_audit.py

Bias audit: analyze model performance across different subgroups.

Groups analyzed:
    1. Object size (bbox area): small / medium / large
    2. Expression length (word count): short / medium / long
    3. Object aspect ratio: tall / square / wide

Uses existing predictions from results/main/predictions.json
and results/ablation/predictions.json.

Output:
    results/bias_audit.json     — per-group metrics
    results/bias_audit.png      — visualization

Usage:
    python scripts/bias_audit.py
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def compute_iou(pred, gt):
    """Compute IoU for single [xc, yc, w, h] pair."""
    p_x1 = pred[0] - pred[2] / 2
    p_y1 = pred[1] - pred[3] / 2
    p_x2 = pred[0] + pred[2] / 2
    p_y2 = pred[1] + pred[3] / 2

    g_x1 = gt[0] - gt[2] / 2
    g_y1 = gt[1] - gt[3] / 2
    g_x2 = gt[0] + gt[2] / 2
    g_y2 = gt[1] + gt[3] / 2

    ix1 = max(p_x1, g_x1)
    iy1 = max(p_y1, g_y1)
    ix2 = min(p_x2, g_x2)
    iy2 = min(p_y2, g_y2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    pa = pred[2] * pred[3]
    ga = gt[2] * gt[3]
    union = pa + ga - inter

    return inter / max(union, 1e-6)


def categorize_size(bbox):
    """Categorize bbox area as small/medium/large."""
    area = bbox[2] * bbox[3]
    if area < 0.05:
        return "small"
    elif area < 0.20:
        return "medium"
    else:
        return "large"


def categorize_text_length(text):
    """Categorize expression by word count."""
    n = len(text.split())
    if n <= 3:
        return "short (1-3)"
    elif n <= 6:
        return "medium (4-6)"
    else:
        return "long (7+)"


def categorize_aspect_ratio(bbox):
    """Categorize by aspect ratio."""
    w, h = bbox[2], bbox[3]
    if h < 1e-6 or w < 1e-6:
        return "degenerate"
    ratio = w / h
    if ratio < 0.7:
        return "tall"
    elif ratio < 1.4:
        return "square"
    else:
        return "wide"


def analyze_model(predictions, model_name):
    """Compute per-group metrics for one model."""
    groups = {
        "object_size": {},
        "expression_length": {},
        "aspect_ratio": {},
    }

    for pred in predictions:
        gt = pred["gt_bbox"]
        pr = pred["pred_bbox"]
        text = pred["text"]
        iou = compute_iou(pr, gt)

        size_cat = categorize_size(gt)
        text_cat = categorize_text_length(text)
        ar_cat = categorize_aspect_ratio(gt)

        for group_name, cat in [
            ("object_size", size_cat),
            ("expression_length", text_cat),
            ("aspect_ratio", ar_cat),
        ]:
            if cat not in groups[group_name]:
                groups[group_name][cat] = []
            groups[group_name][cat].append(iou)

    results = {}
    for group_name, cats in groups.items():
        results[group_name] = {}
        for cat, ious in sorted(cats.items()):
            results[group_name][cat] = {
                "count": len(ious),
                "mean_iou": round(float(np.mean(ious)), 4),
                "median_iou": round(float(np.median(ious)), 4),
                "std_iou": round(float(np.std(ious)), 4),
            }

    return results


def plot_bias_audit(audit_results, output_path):
    """Create bias audit visualization."""
    models = list(audit_results.keys())
    group_names = list(audit_results[models[0]].keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"main": "#4CAF50", "ablation": "#FF9800"}
    bar_width = 0.35

    for idx, group_name in enumerate(group_names):
        ax = axes[idx]
        categories = sorted(audit_results[models[0]][group_name].keys())
        x = np.arange(len(categories))

        for i, model in enumerate(models):
            means = [audit_results[model][group_name][c]["mean_iou"] for c in categories]
            counts = [audit_results[model][group_name][c]["count"] for c in categories]
            bars = ax.bar(x + i * bar_width, means, bar_width,
                          label=model, color=colors.get(model, "#999"),
                          edgecolor="black", linewidth=0.5)
            for bar, val, cnt in zip(bars, means, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}\n(n={cnt})", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(categories, fontsize=9)
        title = group_name.replace("_", " ").title()
        ax.set_title(f"IoU by {title}")
        ax.set_ylabel("Mean IoU")
        ax.set_ylim(0, max(0.55, ax.get_ylim()[1] * 1.15))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Bias Audit — Model Performance by Subgroup", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    audit_results = {}

    for model_name in ["main", "ablation"]:
        pred_path = os.path.join(RESULTS_DIR, model_name, "predictions.json")
        if not os.path.exists(pred_path):
            print(f"Skipping {model_name}: {pred_path} not found")
            continue
        with open(pred_path) as f:
            predictions = json.load(f)
        print(f"Analyzing {model_name}: {len(predictions)} predictions")
        audit_results[model_name] = analyze_model(predictions, model_name)

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "bias_audit.json")
    with open(out_json, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"Saved: {out_json}")

    # Print summary
    for model, groups in audit_results.items():
        print(f"\n{'='*50}")
        print(f"  {model.upper()}")
        print(f"{'='*50}")
        for group_name, cats in groups.items():
            print(f"\n  {group_name}:")
            for cat, metrics in cats.items():
                print(f"    {cat:15s}  IoU={metrics['mean_iou']:.4f}  "
                      f"(n={metrics['count']}, std={metrics['std_iou']:.4f})")

    # Plot
    out_png = os.path.join(RESULTS_DIR, "bias_audit.png")
    plot_bias_audit(audit_results, out_png)


if __name__ == "__main__":
    main()
