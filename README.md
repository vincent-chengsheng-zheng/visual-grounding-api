# visual-grounding-api — Handover Document
> For: Next developer taking over enterprise repo
> Date: Apr 12, 2026
> Deadline: **Apr 17, 2026 (11:59pm)**
> Course: 61.502 Deep Learning for Enterprise

---

## Context

This repo is the **enterprise version** of the GenAI project (`spatial-llava` / `multimodal-grounding-research`).

The core research is already done — models trained, evaluated, results saved. Your job is to:
1. Set up FastAPI deployment
2. Add Docker support
3. Generate training curves
4. Write the enterprise report (10-15 pages)

---

## Where to Find Everything

### Trained Model Weights (Google Drive)
Checkpoints are too large for GitHub. Download from Google Drive:
- `checkpoints/main/best.pth` — 98MB (LoRA + MLP head, IoU=0.386)
- `checkpoints/ablation/best.pth` — 26MB (MLP head only, IoU=0.284)

> **TODO: Insert Google Drive link here after upload**

Place them at:
```
visual-grounding-api/
└── checkpoints/
    ├── main/best.pth
    └── ablation/best.pth
```

### Cluster Access
```
Host : login2.gpucluster.sutd.edu.sg:8888
User : jovyan
GPU  : A100-SXM4-80GB

# Repo also lives on cluster at:
/tmp/SPATIAL-LLAVA/        ← active (clears on restart!)
~/SPATIAL-LLAVA_APR10/     ← backup (persistent)
```

### Results Already Done
All evaluation results are in the repo:
```
results/evaluation/comparison.json   ← final test metrics, all 3 models
results/main/metrics.json            ← training history (loss curves)
results/ablation/metrics.json        ← training history
results/main/predictions.json        ← all test predictions
results/ablation/predictions.json    ← all test predictions
results/*/examples/*.png             ← qualitative prediction images
```

### Key Numbers (for report)
| Model | Val IoU | Test IoU | RMSE | MAE |
|---|---|---|---|---|
| Baseline | 0.097 | 0.097 | 0.288 | 0.238 |
| Ablation | 0.267 | 0.284 | 0.224 | 0.177 |
| Main | 0.357 | 0.386 | 0.172 | 0.119 |

---

## Repo Structure (Target)

```
visual-grounding-api/
├── core/                          ← model, data, loss, utils (DO NOT TOUCH)
├── pipeline/                      ← training + eval scripts (DO NOT TOUCH)
├── results/                       ← all metrics and predictions (DO NOT TOUCH)
├── logs/                          ← training logs (DO NOT TOUCH)
├── demo/
│   └── demo_gradio.py             ← Gradio demo (already working)
├── api/                           ← NEW: FastAPI server
│   ├── __init__.py
│   ├── server.py                  ← main FastAPI app
│   ├── schemas.py                 ← request/response models
│   └── inference.py               ← model loading + inference logic
├── infrastructure/
│   └── docker/
│       ├── Dockerfile             ← NEW: production Dockerfile
│       └── docker-compose.yml     ← NEW: compose file
├── notebooks/
│   └── training_curves.ipynb      ← NEW: plot training curves from metrics.json
├── shared/
│   └── scripts/
│       ├── setup_cluster.sh       ← cluster setup
│       └── download_data.sh       ← data download
├── data/                          ← pkl files (git-tracked)
├── checkpoints/                   ← NOT git-tracked, download from Google Drive
├── weights/                       ← NOT git-tracked, HuggingFace cache
├── requirements.txt               ← already updated
├── README.md                      ← needs enterprise-specific rewrite
└── report/
    └── report.pdf                 ← final submitted report
```

---

## What You Need to Build

### Priority 1: FastAPI Server (core requirement)
File: `api/server.py`

The API should expose:
```
POST /predict
  Input:  { image: base64, text: str, model: "baseline"|"ablation"|"main" }
  Output: { bbox: [xc, yc, w, h], model: str, inference_time_ms: float }

GET /health
  Output: { status: "ok", models_loaded: bool }

GET /models
  Output: { models: ["baseline", "ablation", "main"], metrics: {...} }
```

**How to load models** — copy from `demo/demo_gradio.py`:
```python
from core.model.spatial_llava import load_model
from core.model.llava import StandardLLaVA
from core.paths import PATHS
import torch

# Load on startup (once)
ablation_model, ablation_processor = load_model(use_lora=False, device='cuda')
ckpt = torch.load('checkpoints/ablation/best.pth', map_location='cuda')
ablation_model.load_state_dict(ckpt['model_state'], strict=False)
ablation_model.eval()

main_model, main_processor = load_model(use_lora=True, device='cuda')
ckpt = torch.load('checkpoints/main/best.pth', map_location='cuda')
main_model.load_state_dict(ckpt['model_state'], strict=False)
main_model.eval()
```

**How to run inference on one image** — copy from `demo/demo_gradio.py`:
```python
from PIL import Image
import base64, io

def predict_single(image_b64: str, text: str, model, processor, device='cuda'):
    # Decode base64 image
    img_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    from core.data.preprocessing import PROMPT_TEMPLATE
    prompt = PROMPT_TEMPLATE.format(text=text)
    inputs = processor(
        text=[prompt], images=[image],
        return_tensors='pt', padding=True,
    ).to(device)
    with torch.no_grad():
        pred = model(
            inputs['input_ids'],
            inputs['attention_mask'],
            inputs['pixel_values'],
        )
    return pred[0].cpu().tolist()  # [xc, yc, w, h]
```

Run server:
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Priority 2: Docker
File: `infrastructure/docker/Dockerfile`

Base image: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`

Steps:
1. Copy repo
2. Install requirements
3. Download weights (or mount volume)
4. Expose port 8000
5. CMD: `uvicorn api.server:app --host 0.0.0.0 --port 8000`

### Priority 3: Training Curves
File: `notebooks/training_curves.ipynb`

Plot from `results/main/metrics.json` and `results/ablation/metrics.json`:
- Training loss per epoch (L1 + GIoU)
- Validation IoU per epoch
- Side-by-side comparison: ablation vs main

```python
import json, matplotlib.pyplot as plt

with open('results/main/metrics.json') as f:
    main_metrics = json.load(f)
with open('results/ablation/metrics.json') as f:
    ablation_metrics = json.load(f)

# metrics.json contains per-epoch train_loss, val_iou etc.
# Plot and save as results/training_curves.png
```

### Priority 4: README (enterprise version)
Must include:
- How to download checkpoints from Google Drive
- How to run FastAPI server
- How to build Docker image
- Exact commands to reproduce all results
- Package dependencies

---

## Course Requirements Checklist (61.502 Enterprise)

| Requirement | Status | Notes |
|---|---|---|
| Deep learning model | ✅ done | LLaVA + LoRA + MLP head |
| Custom dataset | ✅ done | RefCOCO (42k samples) |
| Train/val/test split | ✅ done | 20k/1k/1975 |
| Baseline comparison | ✅ done | 3 models |
| RMSE + MAE metrics | ✅ done | in results/ |
| Well-structured repo | ✅ done | |
| requirements.txt | ✅ done | |
| FastAPI deployment | ⏳ TODO | Priority 1 |
| Docker | ⏳ TODO | Priority 2 |
| Training curves | ⏳ TODO | Priority 3 |
| Pipeline diagram | ⏳ TODO | add to report |
| Failure case examples | ⏳ TODO | use results/*/examples/ |
| Google Drive weights | ⏳ TODO | upload checkpoints |
| Report (10-15 pages) | ⏳ TODO | Priority 4 |
| GitHub public | ✅ done | |

---

## Report Structure (10-15 pages)

1. **Executive Summary** (1 page max)
   - Problem: visual grounding via LLaVA + MLP head
   - Result: IoU improved from 0.097 → 0.386 (+297.5%)

2. **Background & Introduction**
   - Visual grounding problem
   - Why LLaVA needs modification
   - RefCOCO dataset

3. **Related Work**
   - LLaVA-1.5 (Liu et al., NeurIPS 2023)
   - u-LLaVA (Xu et al., ECAI 2024)
   - PixelLLM (Xu et al., CVPR 2024)
   - GIoU Loss (Rezatofighi et al., CVPR 2019)

4. **Problem Formulation**
   - Input: image + referring expression
   - Output: [xc, yc, w, h] normalised bbox
   - Architecture diagram (Image → LLaVA → [LOC] token → MLP → bbox)

5. **Data Description**
   - RefCOCO: 42,404 train / 3,811 val / 1,975 test
   - Data stats in `data/dataset_stats.json`
   - Preprocessing: prompt template, image resizing

6. **Method**
   - Baseline: regex parsing
   - Ablation: frozen backbone + MLP head
   - Main: LoRA (rank=16) + MLP head
   - Loss: L1 + GIoU
   - Training config: lr=2e-4, batch=8, epochs=10

7. **Experiments & Results**
   - Results table (copy from above)
   - Training curves (from notebooks/training_curves.ipynb)
   - Qualitative examples (from results/*/examples/)

8. **Failure Cases**
   - Show examples where model fails
   - Discussion of why (small objects, ambiguous text, etc.)

9. **Deployment**
   - FastAPI server description
   - Docker setup
   - How to run

10. **Conclusion & Future Work**
    - Limitations
    - Future: larger LoRA rank, more data, DETR head

---

## Cluster Setup (if you need to retrain or run inference)

```bash
# If /tmp/ was cleared:
git clone https://github.com/vincent-chengsheng-zheng/visual-grounding-api /tmp/SPATIAL-LLAVA
cd /tmp/SPATIAL-LLAVA

# Install dependencies
pip install flash-attn --no-cache-dir --no-build-isolation
pip install -r requirements.txt --break-system-packages

# Set offline mode (weights already downloaded on cluster)
export TRANSFORMERS_OFFLINE=1

# Copy checkpoints from backup
cp -r ~/SPATIAL-LLAVA_APR10/checkpoints /tmp/SPATIAL-LLAVA/
```

---

## Group Member Contributions

| Member | Contribution |
|---|---|
| Vincent (you) | Architecture design, training pipeline, evaluation, Gradio demo, repo setup |
| Member 2 | Gradio demo UI, testing |
| Member 3 | Data preparation, training experiments |
| Member 4 | Report writing, presentation slides |

> Update this table with actual names before submission.

---

## References

- LLaVA-1.5: https://arxiv.org/abs/2310.03744
- u-LLaVA: https://arxiv.org/abs/2311.05348
- PixelLLM: https://arxiv.org/abs/2312.09237
- GIoU: https://arxiv.org/abs/1902.09630
- RefCOCO: https://arxiv.org/abs/1608.00272