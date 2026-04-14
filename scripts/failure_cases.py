"""
scripts/failure_cases.py

Extract and visualize the worst-performing predictions (model malfunctions).

Selects the N samples with lowest IoU, draws pred (red) vs GT (green) boxes,
and saves annotated images. Also categorizes failure modes.

Output:
    results/failure_cases/
        worst_01.png ... worst_10.png   — visualizations
        failure_analysis.json           — per-sample IoU + failure category
        failure_summary.png             — summary chart

Usage:
    python scripts/failure_cases.py
"""

import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_WORST = 10
N_BEST = 5


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


def categorize_failure(pred_bbox, gt_bbox, text):
    """Classify why the model might have failed."""
    gt_area = gt_bbox[2] * gt_bbox[3]
    iou = compute_iou(pred_bbox, gt_bbox)

    if gt_area < 0.03:
        return "small object"

    # Check if prediction is roughly correct location but wrong size
    dist = ((pred_bbox[0] - gt_bbox[0])**2 + (pred_bbox[1] - gt_bbox[1])**2)**0.5
    if dist < 0.15 and iou < 0.3:
        return "correct location, wrong scale"

    if dist > 0.3:
        return "wrong location"

    if len(text.split()) >= 7:
        return "complex expression"

    return "ambiguous reference"


def draw_boxes(image_path, pred_bbox, gt_bbox, text, iou, failure_cat, output_path):
    """Draw pred (red) and GT (green) boxes on image with annotation."""
    # Remap path if needed
    if not os.path.exists(image_path):
        image_path = image_path.replace("/tmp/SPATIAL-LLAVA", "/home/cj/visual-grounding-api")
    if not os.path.exists(image_path):
        print(f"  Image not found: {image_path}")
        return False

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # GT box (green)
    gx1 = int((gt_bbox[0] - gt_bbox[2] / 2) * w)
    gy1 = int((gt_bbox[1] - gt_bbox[3] / 2) * h)
    gx2 = int((gt_bbox[0] + gt_bbox[2] / 2) * w)
    gy2 = int((gt_bbox[1] + gt_bbox[3] / 2) * h)
    draw.rectangle([gx1, gy1, gx2, gy2], outline="green", width=3)

    # Pred box (red)
    px1 = int((pred_bbox[0] - pred_bbox[2] / 2) * w)
    py1 = int((pred_bbox[1] - pred_bbox[3] / 2) * h)
    px2 = int((pred_bbox[0] + pred_bbox[2] / 2) * w)
    py2 = int((pred_bbox[1] + pred_bbox[3] / 2) * h)
    draw.rectangle([px1, py1, px2, py2], outline="red", width=3)

    # Label
    label = f'"{text}" | IoU={iou:.3f} | {failure_cat}'
    draw.text((10, 10), label, fill="white")
    draw.text((10, 25), "Green=GT  Red=Pred", fill="white")

    img.save(output_path)
    return True


def main():
    out_dir = os.path.join(RESULTS_DIR, "failure_cases")
    os.makedirs(out_dir, exist_ok=True)

    pred_path = os.path.join(RESULTS_DIR, "main", "predictions.json")
    with open(pred_path) as f:
        predictions = json.load(f)

    # Compute IoU for each prediction
    for p in predictions:
        p["iou"] = compute_iou(p["pred_bbox"], p["gt_bbox"])
        p["failure_category"] = categorize_failure(p["pred_bbox"], p["gt_bbox"], p["text"])

    # Sort by IoU
    sorted_preds = sorted(predictions, key=lambda x: x["iou"])

    # Save worst cases
    print(f"\n{'='*60}")
    print(f"  WORST {N_WORST} PREDICTIONS (Main Model)")
    print(f"{'='*60}")

    worst_cases = []
    for i, p in enumerate(sorted_preds[:N_WORST]):
        out_path = os.path.join(out_dir, f"worst_{i+1:02d}.png")
        success = draw_boxes(
            p["image_path"], p["pred_bbox"], p["gt_bbox"],
            p["text"], p["iou"], p["failure_category"], out_path
        )
        status = "saved" if success else "FAILED"
        print(f"  #{i+1}  IoU={p['iou']:.4f}  [{p['failure_category']}]  "
              f'"{p["text"]}"  ({status})')
        worst_cases.append({
            "rank": i + 1,
            "text": p["text"],
            "iou": round(p["iou"], 4),
            "failure_category": p["failure_category"],
            "gt_bbox": p["gt_bbox"],
            "pred_bbox": p["pred_bbox"],
            "image_path": p["image_path"],
        })

    # Save best cases for contrast
    print(f"\n{'='*60}")
    print(f"  BEST {N_BEST} PREDICTIONS (Main Model)")
    print(f"{'='*60}")

    best_cases = []
    for i, p in enumerate(sorted_preds[-N_BEST:]):
        out_path = os.path.join(out_dir, f"best_{i+1:02d}.png")
        draw_boxes(
            p["image_path"], p["pred_bbox"], p["gt_bbox"],
            p["text"], p["iou"], "success", out_path
        )
        print(f"  #{i+1}  IoU={p['iou']:.4f}  \"{p['text']}\"")
        best_cases.append({
            "rank": i + 1,
            "text": p["text"],
            "iou": round(p["iou"], 4),
        })

    # Failure category summary
    categories = {}
    for p in predictions:
        cat = p["failure_category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "ious": []}
        categories[cat]["count"] += 1
        categories[cat]["ious"].append(p["iou"])

    category_summary = {}
    for cat, data in sorted(categories.items()):
        category_summary[cat] = {
            "count": data["count"],
            "mean_iou": round(float(np.mean(data["ious"])), 4),
            "pct_of_total": round(data["count"] / len(predictions) * 100, 1),
        }

    print(f"\n{'='*60}")
    print(f"  FAILURE CATEGORY DISTRIBUTION")
    print(f"{'='*60}")
    for cat, stats in sorted(category_summary.items(), key=lambda x: -x[1]["count"]):
        print(f"  {cat:30s}  n={stats['count']:4d}  ({stats['pct_of_total']:5.1f}%)  "
              f"mean IoU={stats['mean_iou']:.4f}")

    # Save analysis JSON
    analysis = {
        "model": "main",
        "total_predictions": len(predictions),
        "worst_cases": worst_cases,
        "best_cases": best_cases,
        "failure_categories": category_summary,
        "overall_mean_iou": round(float(np.mean([p["iou"] for p in predictions])), 4),
    }
    out_json = os.path.join(out_dir, "failure_analysis.json")
    with open(out_json, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Failure category bar chart
    ax = axes[0]
    cats = sorted(category_summary.keys())
    counts = [category_summary[c]["count"] for c in cats]
    mean_ious = [category_summary[c]["mean_iou"] for c in cats]
    bars = ax.barh(cats, counts, color=["#F44336" if m < 0.3 else "#FF9800" if m < 0.4 else "#4CAF50"
                                         for m in mean_ious], edgecolor="black", linewidth=0.5)
    for bar, cnt, miou in zip(bars, counts, mean_ious):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"n={cnt}, IoU={miou:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title("Failure Category Distribution")
    ax.grid(True, alpha=0.3, axis="x")

    # IoU distribution of worst vs rest
    ax = axes[1]
    all_ious = [p["iou"] for p in predictions]
    ax.hist(all_ious, bins=30, color="#2196F3", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="IoU=0.5 threshold")
    ax.axvline(x=np.mean(all_ious), color="green", linestyle="--", alpha=0.7,
               label=f"Mean IoU={np.mean(all_ious):.3f}")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title("IoU Distribution (Main Model, Val Set)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Failure Analysis — Main Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_summary = os.path.join(RESULTS_DIR, "failure_summary.png")
    plt.savefig(out_summary, bbox_inches="tight")
    print(f"Saved: {out_summary}")


if __name__ == "__main__":
    main()
