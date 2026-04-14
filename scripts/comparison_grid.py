"""
scripts/comparison_grid.py

Generate side-by-side comparison grids: Main vs Ablation on same images.
Picks a mix of good, medium, and bad predictions to show range of performance.

Output:
    results/comparison_grid.png

Usage:
    python scripts/comparison_grid.py
"""

import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 9, "figure.dpi": 150})

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


def draw_on_image(image_path, pred_bbox, gt_bbox, pred_color="red"):
    """Draw pred and GT boxes, return PIL image."""
    if not os.path.exists(image_path):
        image_path = image_path.replace("/tmp/SPATIAL-LLAVA", "/home/cj/visual-grounding-api")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # GT (green)
    gx1 = int((gt_bbox[0] - gt_bbox[2] / 2) * w)
    gy1 = int((gt_bbox[1] - gt_bbox[3] / 2) * h)
    gx2 = int((gt_bbox[0] + gt_bbox[2] / 2) * w)
    gy2 = int((gt_bbox[1] + gt_bbox[3] / 2) * h)
    draw.rectangle([gx1, gy1, gx2, gy2], outline="lime", width=3)

    # Pred
    px1 = int((pred_bbox[0] - pred_bbox[2] / 2) * w)
    py1 = int((pred_bbox[1] - pred_bbox[3] / 2) * h)
    px2 = int((pred_bbox[0] + pred_bbox[2] / 2) * w)
    py2 = int((pred_bbox[1] + pred_bbox[3] / 2) * h)
    draw.rectangle([px1, py1, px2, py2], outline=pred_color, width=3)

    return img


def main():
    with open(os.path.join(RESULTS_DIR, "main", "predictions.json")) as f:
        main_preds = json.load(f)
    with open(os.path.join(RESULTS_DIR, "ablation", "predictions.json")) as f:
        abl_preds = json.load(f)

    # Compute IoUs
    for m, a in zip(main_preds, abl_preds):
        m["iou"] = compute_iou(m["pred_bbox"], m["gt_bbox"])
        a["iou"] = compute_iou(a["pred_bbox"], a["gt_bbox"])

    # Select 6 samples: 2 where main is much better, 2 medium, 2 where both struggle
    paired = list(zip(main_preds, abl_preds))
    paired.sort(key=lambda x: x[0]["iou"] - x[1]["iou"], reverse=True)

    # Main wins big
    main_wins = [(m, a) for m, a in paired if m["iou"] > 0.5 and a["iou"] < 0.3][:2]
    # Both decent
    both_ok = [(m, a) for m, a in paired if 0.3 < m["iou"] < 0.6 and 0.2 < a["iou"] < 0.5]
    both_ok = both_ok[len(both_ok)//2:len(both_ok)//2+2]  # middle of range
    # Both fail
    both_fail = [(m, a) for m, a in paired if m["iou"] < 0.15 and a["iou"] < 0.15][:2]

    selected = main_wins + both_ok + both_fail
    if len(selected) < 6:
        # Fallback: just pick spread
        indices = np.linspace(0, len(paired) - 1, 6, dtype=int)
        selected = [paired[i] for i in indices]

    # Create grid: 6 rows x 2 cols (main, ablation)
    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.5 * n_rows))

    for row, (main_p, abl_p) in enumerate(selected):
        # Main
        ax = axes[row][0]
        img_main = draw_on_image(main_p["image_path"], main_p["pred_bbox"], main_p["gt_bbox"], "red")
        ax.imshow(img_main)
        ax.set_title(f'Main (IoU={main_p["iou"]:.3f})', fontsize=10,
                     color="green" if main_p["iou"] > 0.5 else "orange" if main_p["iou"] > 0.25 else "red")
        ax.set_ylabel(f'"{main_p["text"]}"', fontsize=8, rotation=0, labelpad=80, va="center")
        ax.set_xticks([])
        ax.set_yticks([])

        # Ablation
        ax = axes[row][1]
        img_abl = draw_on_image(abl_p["image_path"], abl_p["pred_bbox"], abl_p["gt_bbox"], "orange")
        ax.imshow(img_abl)
        ax.set_title(f'Ablation (IoU={abl_p["iou"]:.3f})', fontsize=10,
                     color="green" if abl_p["iou"] > 0.5 else "orange" if abl_p["iou"] > 0.25 else "red")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Main vs Ablation — Same Images (Green=GT, Red/Orange=Pred)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "comparison_grid.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
