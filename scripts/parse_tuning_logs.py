"""
scripts/parse_tuning_logs.py

Parse hyperparameter tuning evidence from existing training logs.
Extracts configs and final metrics from all step2/step3 logs.

Output:
    results/hyperparam_tuning.json  — comparison table
    results/hyperparam_tuning.png   — visualization

Usage:
    python scripts/parse_tuning_logs.py
"""

import json
import os
import re
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def parse_log(log_path):
    """Extract config and final val metrics from a training log."""
    with open(log_path) as f:
        content = f.read()

    # Extract config JSON
    config_match = re.search(r"Config:\s*(\{.*?\})", content, re.DOTALL)
    if not config_match:
        return None

    try:
        config = json.loads(config_match.group(1))
    except json.JSONDecodeError:
        return None

    # Extract all val metrics
    val_lines = re.findall(
        r"\[Val\]\s*epoch=(\d+)\s+loss=([\d.]+)\s+IoU=([\d.]+)\s+RMSE=([\d.]+)\s+MAE=([\d.]+)",
        content
    )

    if not val_lines:
        # Try simpler format (step3 / older logs)
        val_lines = re.findall(
            r"\[Val\]\s*epoch=(\d+)\s+loss=([\d.]+)\s+IoU=([\d.]+)",
            content
        )

    if not val_lines:
        return None

    # Take the best val IoU
    best_iou = 0
    best_epoch = 0
    final_metrics = {}
    for line in val_lines:
        epoch = int(line[0])
        iou = float(line[2])
        if iou > best_iou:
            best_iou = iou
            best_epoch = epoch
            final_metrics = {
                "epoch": epoch,
                "val_loss": float(line[1]),
                "val_iou": iou,
            }
            if len(line) > 3:
                final_metrics["val_rmse"] = float(line[3])
                final_metrics["val_mae"] = float(line[4])

    completed = "Step 2 complete" in content or "Step 3 complete" in content

    return {
        "log_file": os.path.basename(log_path),
        "config": config,
        "epochs_logged": len(val_lines),
        "completed": completed,
        "best_val_iou": best_iou,
        "best_epoch": best_epoch,
        "final_metrics": final_metrics,
    }


def main():
    results = []

    for fname in sorted(os.listdir(LOGS_DIR)):
        if not fname.endswith(".log"):
            continue
        path = os.path.join(LOGS_DIR, fname)
        parsed = parse_log(path)
        if parsed and parsed["epochs_logged"] > 0:
            results.append(parsed)

    # Also check SPATIAL-LLAVA logs
    alt_logs = os.path.expanduser("~/SPATIAL-LLAVA/logs")
    if os.path.isdir(alt_logs):
        for fname in sorted(os.listdir(alt_logs)):
            if not fname.endswith(".log"):
                continue
            path = os.path.join(alt_logs, fname)
            parsed = parse_log(path)
            if parsed and parsed["epochs_logged"] > 0:
                parsed["log_file"] = f"SPATIAL-LLAVA/{fname}"
                results.append(parsed)

    # Filter to completed multi-epoch runs
    completed_runs = [r for r in results if r["completed"] and r["epochs_logged"] >= 2]

    # Build comparison table
    comparison = []
    for r in completed_runs:
        c = r["config"]
        comparison.append({
            "log": r["log_file"],
            "epochs": c.get("epochs", r["epochs_logged"]),
            "batch_size": c.get("batch_size", "?"),
            "lr": c.get("lr", "?"),
            "lora_rank": c.get("lora_rank", "?"),
            "weight_decay": c.get("weight_decay", 0),
            "max_length": c.get("max_length", "?"),
            "best_val_iou": r["best_val_iou"],
            "best_epoch": r["best_epoch"],
            "completed": r["completed"],
            "final_metrics": r["final_metrics"],
        })

    # Sort by best IoU descending
    comparison.sort(key=lambda x: -x["best_val_iou"])

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "hyperparam_tuning.json")
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved: {out_json}")

    # Print table
    print(f"\n{'='*90}")
    print(f"  HYPERPARAMETER TUNING RESULTS ({len(comparison)} completed runs)")
    print(f"{'='*90}")
    print(f"  {'Log':<35s} {'EP':>3s} {'BS':>3s} {'LR':>8s} {'WD':>6s} {'MaxLen':>6s} {'Best IoU':>9s} {'@Ep':>4s}")
    print(f"  {'-'*35} {'-'*3} {'-'*3} {'-'*8} {'-'*6} {'-'*6} {'-'*9} {'-'*4}")
    for r in comparison:
        print(f"  {r['log']:<35s} {r['epochs']:>3} {r['batch_size']:>3} "
              f"{r['lr']:>8.1e} {r['weight_decay']:>6.3f} {str(r['max_length']):>6s} "
              f"{r['best_val_iou']:>9.4f} {r['best_epoch']:>4}")

    # Plot
    if len(comparison) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of best IoU per run
        ax = axes[0]
        labels = []
        ious = []
        colors_list = []
        for r in comparison:
            lr_str = f"{r['lr']:.0e}" if isinstance(r['lr'], float) else str(r['lr'])
            label = f"bs={r['batch_size']}\nlr={lr_str}\nwd={r['weight_decay']}"
            labels.append(label)
            ious.append(r["best_val_iou"])
            # Highlight the best
            colors_list.append("#4CAF50" if r["best_val_iou"] == max(c["best_val_iou"] for c in comparison) else "#90CAF9")

        x = np.arange(len(labels))
        bars = ax.bar(x, ious, color=colors_list, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, ious):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Best Val IoU")
        ax.set_title("Hyperparameter Comparison")
        ax.grid(True, alpha=0.3, axis="y")

        # LR vs IoU scatter
        ax = axes[1]
        lrs = [r["lr"] for r in comparison if isinstance(r["lr"], float)]
        lr_ious = [r["best_val_iou"] for r in comparison if isinstance(r["lr"], float)]
        bss = [r["batch_size"] for r in comparison if isinstance(r["lr"], float)]
        for lr_val, iou_val, bs in zip(lrs, lr_ious, bss):
            ax.scatter(lr_val, iou_val, s=100, edgecolor="black", linewidth=0.5, zorder=5)
            ax.annotate(f"bs={bs}", (lr_val, iou_val), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Best Val IoU")
        ax.set_title("Learning Rate vs Performance")
        ax.grid(True, alpha=0.3)

        plt.suptitle("Hyperparameter Tuning Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_png = os.path.join(RESULTS_DIR, "hyperparam_tuning.png")
        plt.savefig(out_png, bbox_inches="tight")
        print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
