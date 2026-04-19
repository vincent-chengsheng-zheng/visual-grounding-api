"""
pipeline/step5_hyperparam_tuning.py

Hyperparameter tuning: train multiple configs and compare.
Saves each run to results/tuning/<run_name>/ without touching existing checkpoints.

Configs tested:
    - rank_8:  LoRA rank=8,  lr=2e-4  (less capacity)
    - rank_32: LoRA rank=32, lr=2e-4  (more capacity)
    - lr_1e4:  LoRA rank=16, lr=1e-4  (lower learning rate)

The original main model (rank=16, lr=2e-4) serves as the reference.

How to run:
    python pipeline/step5_hyperparam_tuning.py

    # Single config only:
    python pipeline/step5_hyperparam_tuning.py --configs rank_8

    # Quick smoke test:
    python pipeline/step5_hyperparam_tuning.py --max_samples 200 --epochs 3
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.paths import PATHS
from core.data.refcoco_loader import make_loaders
from core.model.spatial_llava import load_model
from core.loss.spatial_loss import SpatialLoss
from core.utils.metrics import compute_all_metrics


TUNING_CONFIGS = {
    "rank_8": {"lora_rank": 8, "lr": 2e-4},
    "rank_32": {"lora_rank": 32, "lr": 2e-4},
    "lr_1e4": {"lora_rank": 16, "lr": 1e-4},
}

FIXED = {
    "epochs": 10,
    "batch_size": 8,
    "weight_decay": 0.0,
    "max_length": 128,
    "num_workers": 4,
    "seed": 42,
}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        pix = batch["pixel_values"].to(device)
        tgt = batch["bbox"].to(device)

        optimizer.zero_grad()
        pred = model(ids, mask, pix)
        loss = criterion(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(pred.detach().cpu())
        all_targets.append(tgt.detach().cpu())

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)
    m = compute_all_metrics(preds_t, targets_t)
    return {"train_loss": total_loss / len(loader), "train_iou": m["mean_iou"],
            "train_rmse": m["rmse"], "train_mae": m["mae"]}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        pix = batch["pixel_values"].to(device)
        tgt = batch["bbox"].to(device)

        pred = model(ids, mask, pix)
        loss = criterion(pred, tgt)
        total_loss += loss.item()
        all_preds.append(pred.cpu())
        all_targets.append(tgt.cpu())

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)
    m = compute_all_metrics(preds_t, targets_t)
    return {"val_loss": total_loss / len(loader), "val_iou": m["mean_iou"],
            "val_rmse": m["rmse"], "val_mae": m["mae"]}


def run_config(name: str, config: dict, args):
    set_seed(FIXED["seed"])
    device = "cuda"
    epochs = args.epochs or FIXED["epochs"]
    lora_rank = config["lora_rank"]
    lr = config["lr"]

    out_dir = PATHS.results / "tuning" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = PATHS.checkpoints / "tuning" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Tuning run: {name}")
    print(f"  LoRA rank={lora_rank}, lr={lr}, epochs={epochs}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    model, processor = load_model(use_lora=True, device=device, lora_rank=lora_rank)

    train_loader, val_loader = make_loaders(
        processor=processor,
        batch_size=FIXED["batch_size"],
        num_workers=FIXED["num_workers"],
        max_length=FIXED["max_length"],
        max_samples=args.max_samples,
    )

    optimizer = AdamW(model.trainable_parameters(), lr=lr, weight_decay=FIXED["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = SpatialLoss(w_l1=1.0, w_giou=1.0)

    history = []
    best_iou = -1.0
    t_start = time.time()

    for epoch in range(epochs):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_m = {"epoch": epoch + 1, "lr": scheduler.get_last_lr()[0],
                   **train_m, **val_m}
        history.append(epoch_m)

        tag = ""
        if val_m["val_iou"] > best_iou:
            best_iou = val_m["val_iou"]
            torch.save({"model_state": model.state_dict(), "config": config,
                         "epoch": epoch, "val_iou": best_iou},
                        str(ckpt_dir / "best.pth"))
            tag = " ★ best"

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"train_IoU={train_m['train_iou']:.4f} | "
              f"val_IoU={val_m['val_iou']:.4f} | "
              f"{elapsed:.0f}s{tag}")

    total_time = time.time() - t_start

    results = {
        "name": name,
        "config": {**FIXED, **config, "epochs": epochs},
        "best_val_iou": best_iou,
        "final_val_iou": history[-1]["val_iou"],
        "final_val_rmse": history[-1]["val_rmse"],
        "final_val_mae": history[-1]["val_mae"],
        "training_time_min": round(total_time / 60, 1),
        "history": history,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ {name} done — best val IoU: {best_iou:.4f} ({total_time/60:.1f} min)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning runs")
    parser.add_argument("--configs", nargs="+", default=list(TUNING_CONFIGS.keys()),
                        choices=list(TUNING_CONFIGS.keys()),
                        help="Which configs to run (default: all)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap train samples")
    args = parser.parse_args()

    all_results = {}

    for name in args.configs:
        config = TUNING_CONFIGS[name]
        result = run_config(name, config, args)
        all_results[name] = {
            "best_val_iou": result["best_val_iou"],
            "final_val_rmse": result["final_val_rmse"],
            "final_val_mae": result["final_val_mae"],
            "training_time_min": result["training_time_min"],
            "config": result["config"],
        }
        # Free GPU memory before next run
        torch.cuda.empty_cache()

    # Add reference (original main model)
    main_metrics_path = PATHS.results_main / "metrics.json"
    if main_metrics_path.exists():
        with open(main_metrics_path) as f:
            main_m = json.load(f)
        all_results["main_reference"] = {
            "best_val_iou": main_m["val_iou"],
            "final_val_rmse": main_m["val_rmse"],
            "final_val_mae": main_m["val_mae"],
            "config": {"lora_rank": 16, "lr": 2e-4, "epochs": 10},
        }

    # Save comparison
    comp_path = PATHS.results / "tuning" / "comparison.json"
    with open(comp_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("  TUNING SUMMARY")
    print(f"{'='*60}")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["best_val_iou"]):
        print(f"  {name:20s} | val IoU: {r['best_val_iou']:.4f} | "
              f"RMSE: {r.get('final_val_rmse', 'N/A')}")
    print(f"\n  Saved: {comp_path}")


if __name__ == "__main__":
    main()
