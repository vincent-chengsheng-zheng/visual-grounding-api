# Spatial-LLaVA: Visual Grounding via Regression-Based Bounding Box Prediction

> **Course:** 61.502 Deep Learning for Enterprise, SUTD (Y2026)  
> **Dataset:** RefCOCO (48,190 samples)  
> **Model:** LLaVA-1.5-7B + LoRA + MLP Regression Head

## Overview

Visual grounding system that localizes objects in images from natural language descriptions. We replace LLaVA's text-based coordinate output with a direct bounding box regression head, eliminating parsing failures and improving accuracy by **+297.5%** over the baseline.

**Key Results (RefCOCO test set, 1,975 samples):**

| Model | IoU | RMSE | MAE | Method |
|---|---|---|---|---|
| Baseline | 0.097 | 0.288 | 0.238 | Vanilla LLaVA + regex parsing |
| Ablation | 0.284 | 0.224 | 0.177 | Frozen LLaVA + MLP head |
| **Main** | **0.386** | **0.172** | **0.119** | **LoRA + MLP head** |

---

## Repository Structure

```
visual-grounding-api/
├── core/                           # Core ML modules
│   ├── model/                      # SpatialLLaVA, StandardLLaVA, RegressionHead, LoRA config
│   ├── data/                       # RefCOCO dataset loaders (tensor + PIL variants)
│   ├── loss/                       # L1 + GIoU spatial loss
│   └── utils/                      # Metrics (IoU, RMSE, MAE), checkpoints, visualization
├── pipeline/                       # Training and evaluation scripts
│   ├── stage_0_environment.py      # Environment health check
│   ├── stage_1_data_preparation.py # Download RefCOCO + COCO, generate pkl files
│   ├── step1_baseline_inference.py # Baseline: vanilla LLaVA + regex
│   ├── step2_train_main.py         # Main: LoRA + MLP head (10 epochs)
│   ├── step3_train_ablation.py     # Ablation: frozen backbone + MLP head (10 epochs)
│   └── step4_evaluate.py           # Final 3-model comparison on test set
├── api/                            # FastAPI server (predict, health, models endpoints)
├── demo/                           # Gradio interactive demo
├── frontend/                       # React web UI (src + production build)
├── scripts/                        # Analysis and visualization
│   ├── bias_audit.py               # Performance by subgroup (size, length, aspect ratio)
│   ├── failure_cases.py            # Worst predictions + failure categories
│   ├── accuracy_at_threshold.py    # Acc@IoU thresholds
│   ├── comparison_grid.py          # Main vs Ablation side-by-side
│   ├── parse_tuning_logs.py        # Extract HP tuning results from logs
│   └── benchmark_latency.py        # API latency measurement
├── notebooks/                      # Training curves + figure reproduction index
├── infrastructure/docker/          # Dockerfile (CUDA 12.8, PyTorch 2.11)
├── shared/scripts/                 # setup_cluster.sh, download_data.sh
├── data/                           # RefCOCO pkl files (tracked) + COCO images (gitignored)
├── results/                        # All metrics, plots, predictions, analysis outputs
├── logs/                           # Training logs (hyperparameter tuning evidence)
├── checkpoints/                    # Trained model weights (not in git, download from Google Drive)
├── .github/workflows/              # CI: lint + test + docker build
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (trained on A100-SXM4-80GB)
- ~15GB disk for COCO images, ~20GB for model weights

### 1. Clone and Install

```bash
git clone https://github.com/vincent-chengsheng-zheng/visual-grounding-api.git
cd visual-grounding-api
pip install -r requirements.txt
pip install flash-attn --no-cache-dir --no-build-isolation
```

### 2. Download Data

```bash
bash shared/scripts/download_data.sh
```

This downloads COCO train2014 images (~13.5GB, 82,783 images) and generates RefCOCO pickle files:
- `data/refcoco_train.pkl` — 42,404 samples
- `data/refcoco_val.pkl` — 3,811 samples
- `data/refcoco_test.pkl` — 1,975 samples

### 3. Download Trained Weights

| Checkpoint | Size | Test IoU | Link |
|---|---|---|---|
| `main_best.pth` (LoRA + MLP head) | 98MB | 0.386 | [Google Drive](https://drive.google.com/file/d/1A_aBxnWJHOu7sH5iG5TADxeMpCwImoaV/view?usp=sharing) |
| `ablation_best.pth` (MLP head only) | 26MB | 0.284 | [Google Drive](https://drive.google.com/file/d/1VthezY5Q1ND5EPLPFe6pF6CJEmCYVGg3/view?usp=sharing) |

Place them at:
```
checkpoints/
├── main/best.pth
└── ablation/best.pth
```

---

## Training from Scratch

All training uses **seed=42** for reproducibility.

### Step 1: Baseline Inference (no training)

```bash
python pipeline/step1_baseline_inference.py
```

Runs vanilla LLaVA-1.5-7B text generation + regex parsing on the test set. Expected IoU ~ 0.097.

### Step 2: Train Main Model (LoRA + MLP Head)

```bash
python pipeline/step2_train_main.py
```

**Default hyperparameters:**

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 8 |
| Learning rate | 2e-4 (cosine annealing, eta_min=1e-6) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA targets | q_proj, v_proj, k_proj |
| Weight decay | 0.0 |
| Gradient clipping | max_norm=1.0 |
| Seed | 42 |

**Custom hyperparameters:**
```bash
python pipeline/step2_train_main.py --epochs 10 --batch_size 8 --lr 2e-4 --lora_rank 16
```

**Resume training:**
```bash
python pipeline/step2_train_main.py --resume
```

**Output:** `checkpoints/main/best.pth`, `results/main/metrics.json`, `results/main/examples/*.png`

### Step 3: Train Ablation Model (Frozen Backbone + MLP Head)

```bash
python pipeline/step3_train_ablation.py
```

Same as Step 2 but with `use_lora=False` — entire LLaVA backbone is frozen, only the MLP head is trained (2.2M params vs 10.2M).

**Output:** `checkpoints/ablation/best.pth`, `results/ablation/metrics.json`

### Step 4: Evaluate All Models on Test Set

```bash
python pipeline/step4_evaluate.py
```

Evaluates baseline, ablation, and main on RefCOCO test split (1,975 samples). **Output:** `results/evaluation/comparison.json`

---

## Reproducing Figures

All figures can be regenerated from saved JSON data without GPU.

### Training curves and model comparison (Figures 1-6):

```bash
# Run the Jupyter notebook
jupyter notebook notebooks/training_curves.ipynb
```

Or run individual sections — the notebook reads from `results/main/metrics.json` and `results/ablation/metrics.json`.

### Analysis plots:

```bash
python scripts/bias_audit.py              # Bias audit (Figure 9)
python scripts/failure_cases.py           # Failure analysis + IoU histogram (Figure 10)
python scripts/accuracy_at_threshold.py   # Accuracy@threshold (Figure 4)
python scripts/comparison_grid.py         # Main vs Ablation comparison (Figure 11)
python scripts/parse_tuning_logs.py       # Hyperparameter tuning summary (Figure 2)
```

**Figure reproduction index** is documented in the final cell of `notebooks/training_curves.ipynb`, listing every figure with its source data and reproduction command.

---

## API Deployment

### FastAPI Server

```bash
export TRANSFORMERS_OFFLINE=1
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Predict bbox for image + text query |
| `/health` | GET | Health check, loaded models, GPU status |
| `/models` | GET | List models with test metrics |

**Example request:**
```json
POST /predict
{
    "image": "<base64-encoded image>",
    "text": "the red chair",
    "model": "main"
}
```

**Response:**
```json
{
    "bbox": [0.45, 0.52, 0.30, 0.40],
    "model": "main",
    "inference_time_ms": 78.5
}
```

**Swagger UI** available at `http://localhost:8000/docs`.

### Gradio Demo

```bash
export TRANSFORMERS_OFFLINE=1
python demo/demo_gradio.py
```

Side-by-side comparison of all 3 models on any uploaded image.

### Docker

```bash
docker build -t visual-grounding-api -f infrastructure/docker/Dockerfile .
docker run --gpus all -p 8000:8000 \
    -v /path/to/weights:/app/weights \
    -v /path/to/checkpoints:/app/checkpoints \
    visual-grounding-api
```

**Inference Latency (A100):**

| Model | GPU (ms) | Roundtrip (ms) |
|---|---|---|
| Baseline | 312.7 | 323.3 |
| Ablation | 73.5 | 86.2 |
| Main | 78.5 | 91.8 |

---

## Model Architecture

```
Image (384x384) ──┐
                   ├──→ LLaVA-1.5-7B ──→ [LOC] Token Hidden State (4096-dim)
Text + [LOC] ─────┘        │                        │
                      LoRA Adapters              MLP Head
                      (rank=16, 8M params)       4096 → 512 → 256 → 4
                                                 GELU + Dropout(0.1)
                                                 Sigmoid → [xc, yc, w, h]
```

**Loss:** L1 + GIoU (both weight 1.0)  
**Trainable parameters:** 10.2M / 7B total (0.14%)

---

## Results Summary

### Test Set Performance

| Model | Test IoU | RMSE | MAE | vs Baseline |
|---|---|---|---|---|
| Baseline | 0.097 | 0.288 | 0.238 | - |
| Ablation | 0.284 | 0.224 | 0.177 | +192.7% |
| **Main** | **0.386** | **0.172** | **0.119** | **+297.5%** |

### Accuracy at IoU Thresholds (Main Model)

| Threshold | Accuracy |
|---|---|
| IoU > 0.10 | 75.8% |
| IoU > 0.25 | 60.1% |
| IoU > 0.50 | 33.1% |
| IoU > 0.75 | 8.3% |

### Bias Audit (Main Model)

| Object Size | IoU | Count |
|---|---|---|
| Small (<5% area) | 0.119 | 16 |
| Medium (5-20%) | 0.296 | 626 |
| Large (>20%) | 0.473 | 358 |

---

## Dependencies

Key packages (full list in `requirements.txt`):

- PyTorch 2.11.0 + CUDA (installed separately)
- `transformers==5.4.0`
- `peft==0.7.1` (LoRA)
- `fastapi==0.115.5`
- `gradio==5.23.3`
- `flash-attn` (separate install)

---

## References

1. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). *Visual Instruction Tuning*. NeurIPS 2023. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
2. Liu, H., Li, C., Li, Y., & Lee, Y. J. (2024). *Improved Baselines with Visual Instruction Tuning*. CVPR 2024. [arXiv:2310.03744](https://arxiv.org/abs/2310.03744)
3. Xu, J., Xu, L., Yang, Y., Li, X., Wang, F., Xie, Y., Huang, Y.-J., & Li, Y. (2024). *u-LLaVA: Unifying Multi-Modal Tasks via Large Language Model*. [arXiv:2311.05348](https://arxiv.org/abs/2311.05348)
4. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
5. Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S. (2019). *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression*. CVPR 2019. [arXiv:1902.09630](https://arxiv.org/abs/1902.09630)
6. Yu, L., Poirson, P., Yang, S., Berg, A. C., & Berg, T. L. (2016). *Modeling Context in Referring Expressions*. ECCV 2016. [arXiv:1608.00272](https://arxiv.org/abs/1608.00272)
