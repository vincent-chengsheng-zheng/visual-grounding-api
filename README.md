# visual-grounding-api — Enterprise Handover
> Course: 61.502 Deep Learning for Enterprise
> Deadline: **Apr 17, 2026 (11:59pm)**
> Last updated: Apr 12, 2026

---

## ⚠️ READ THIS FIRST

The core research is **fully done** — models trained, evaluated, results saved.
Your job is **deployment + report only**. Do not retrain anything.

---

## Trained Weights (Google Drive)

Download before starting anything:

| Checkpoint | Size | IoU | Link |
|---|---|---|---|
| `main_best.pth` (LoRA + MLP head) | 98MB | 0.386 | [Download](https://drive.google.com/file/d/1A_aBxnWJHOu7sH5iG5TADxeMpCwImoaV/view?usp=sharing) |
| `ablation_best.pth` (MLP head only) | 26MB | 0.284 | [Download](https://drive.google.com/file/d/1VthezY5Q1ND5EPLPFe6pF6CJEmCYVGg3/view?usp=sharing) |

Place them at:
```
visual-grounding-api/
└── checkpoints/
    ├── main/best.pth
    └── ablation/best.pth
```

---

## Cluster Access

```
Host : login2.gpucluster.sutd.edu.sg:8888
User : jovyan
GPU  : A100-SXM4-80GB
```

Access via browser at `http://login2.gpucluster.sutd.edu.sg:8888`

Backup of everything on cluster:
```
~/SPATIAL-LLAVA_APR10/
├── checkpoints/main/        ← trained weights (persistent)
├── checkpoints/ablation/    ← trained weights (persistent)
├── results/                 ← all metrics
└── logs/                    ← training logs
```

Cluster setup:
```bash
git clone https://github.com/vincent-chengsheng-zheng/visual-grounding-api /tmp/visual-grounding-api
cd /tmp/visual-grounding-api
pip install flash-attn --no-cache-dir --no-build-isolation
pip install -r requirements.txt --break-system-packages
export TRANSFORMERS_OFFLINE=1
cp -r ~/SPATIAL-LLAVA_APR10/checkpoints /tmp/visual-grounding-api/
```

---

## Key Results (use in report)

Evaluated on RefCOCO test split (1,975 samples):

| Model | Val IoU | Test IoU | RMSE | MAE | Method |
|---|---|---|---|---|---|
| Baseline | 0.097 | 0.097 | 0.288 | 0.238 | Vanilla LLaVA + regex |
| Ablation | 0.267 | 0.284 | 0.224 | 0.177 | Frozen LLaVA + MLP head |
| **Main** | **0.357** | **0.386** | **0.172** | **0.119** | LoRA + MLP head |

Improvements over baseline:
- Ablation: **+192.7%** IoU
- Main: **+297.5%** IoU

All raw metrics in `results/evaluation/comparison.json`.

---

## What You Need to Build (Priority Order)

### 1. FastAPI Server ← most important, do this first

Create `api/server.py`. The API must expose:

```
POST /predict
  Input:  { "image": "<base64>", "text": "the red chair", "model": "main" }
  Output: { "bbox": [xc, yc, w, h], "model": "main", "inference_time_ms": 120 }

GET  /health
  Output: { "status": "ok", "models_loaded": true }

GET  /models
  Output: { "models": ["baseline", "ablation", "main"], "metrics": {...} }
```

**Model loading** (copy from `demo/demo_gradio.py`):
```python
import torch, os, sys
sys.path.insert(0, '.')
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from core.model.spatial_llava import load_model
from core.model.llava import StandardLLaVA
from core.paths import PATHS

DEVICE = 'cuda'

# Ablation
ablation_model, ablation_processor = load_model(use_lora=False, device=DEVICE)
ckpt = torch.load('checkpoints/ablation/best.pth', map_location=DEVICE)
ablation_model.load_state_dict(ckpt['model_state'], strict=False)
ablation_model.eval()

# Main
main_model, main_processor = load_model(use_lora=True, device=DEVICE)
ckpt = torch.load('checkpoints/main/best.pth', map_location=DEVICE)
main_model.load_state_dict(ckpt['model_state'], strict=False)
main_model.eval()
```

**Inference on one image** (copy from `demo/demo_gradio.py`):
```python
from PIL import Image
from core.data.preprocessing import PROMPT_TEMPLATE
import base64, io, time

def predict_single(image_b64: str, text: str, model, processor) -> dict:
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert('RGB')
    prompt = PROMPT_TEMPLATE.format(text=text)
    inputs = processor(
        text=[prompt], images=[img],
        return_tensors='pt', padding=True,
    ).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        pred = model(
            inputs['input_ids'],
            inputs['attention_mask'],
            inputs['pixel_values'],
        )
    ms = (time.time() - t0) * 1000
    return {'bbox': pred[0].cpu().tolist(), 'inference_time_ms': round(ms, 1)}
```

Run server:
```bash
cd /tmp/visual-grounding-api
export TRANSFORMERS_OFFLINE=1
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

### 2. Docker

Create `infrastructure/docker/Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install flash-attn --no-cache-dir --no-build-isolation

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t visual-grounding-api .
docker run --gpus all -p 8000:8000 visual-grounding-api
```

---

### 3. Training Curves

Create `notebooks/training_curves.ipynb`. Plot from existing JSON files:

```python
import json, matplotlib.pyplot as plt

with open('results/main/metrics.json') as f:
    main_m = json.load(f)
with open('results/ablation/metrics.json') as f:
    ablation_m = json.load(f)

# Check what keys are available
print(main_m.keys())

# Plot val IoU per epoch
# Save as results/training_curves.png for report
```

---

### 4. Report (10-15 pages, NeurIPS format NOT required for Enterprise)

Sections required by 61.502:

1. **Executive Summary** (max 1 page)
   - Problem: visual grounding via LLaVA + MLP
   - Result: +297.5% IoU over baseline

2. **Background & Introduction**
   - Visual grounding, why it's hard, why it matters

3. **Related Work**
   - LLaVA-1.5, u-LLaVA, PixelLLM, GIoU loss

4. **Problem Formulation**
   - Input/output definition
   - **Pipeline diagram** ← required by course (draw: Image+Text → LLaVA → [LOC] token → MLP → bbox)

5. **Data Description**
   - RefCOCO: 42,404 train / 3,811 val / 1,975 test
   - Stats in `data/dataset_stats.json`

6. **Method**
   - Baseline (regex), Ablation (frozen + head), Main (LoRA + head)
   - Architecture: 4096 → 512 → 256 → 4 + Sigmoid
   - Loss: L1 + GIoU, lr=2e-4, batch=8, epochs=10

7. **Experiments & Results**
   - Results table (copy from above)
   - **Training curves** ← required (plot from notebooks/training_curves.ipynb)
   - **Qualitative examples** ← use images from `results/*/examples/`

8. **Failure Cases** ← required
   - Show bad predictions from `results/*/examples/`
   - Discuss why: small objects, ambiguous expressions, truncated boxes

9. **Deployment**
   - FastAPI endpoints description
   - Docker setup
   - How to run, latency numbers

10. **Discussion & Recommendations**

11. **Limitations & Future Work**
    - Larger LoRA rank, more training data, DETR-style head

12. **Group Member Contributions** ← required
    | Member | Contribution |
    |---|---|
    | Vincent | Architecture, training pipeline, evaluation, Gradio demo, repo |
    | Member 2 | Gradio demo UI |
    | Member 3 | Data preparation, training experiments |
    | Member 4 | Report, presentation slides |

---

## Course Checklist (61.502)

| Requirement | Status |
|---|---|
| Deep learning model | ✅ done |
| Train/val/test split | ✅ done |
| Baseline comparison | ✅ done |
| RMSE + MAE metrics | ✅ done |
| Well-structured repo + README | ✅ done |
| requirements.txt | ✅ done |
| Weights on Google Drive | ✅ done (links above) |
| GitHub public repo | ✅ done |
| FastAPI deployment | ⏳ **TODO — Priority 1** |
| Docker | ⏳ **TODO — Priority 2** |
| Training curves plot | ⏳ **TODO — Priority 3** |
| Pipeline diagram | ⏳ **TODO — add to report** |
| Failure case analysis | ⏳ **TODO — add to report** |
| Group contributions section | ⏳ **TODO — add to report** |
| Report (10-15 pages) | ⏳ **TODO — Priority 4** |

---

## Repo Structure (Target)

```
visual-grounding-api/
├── core/                          ← DO NOT TOUCH
├── pipeline/                      ← DO NOT TOUCH
├── results/                       ← DO NOT TOUCH
├── logs/                          ← DO NOT TOUCH
├── demo/
│   └── demo_gradio.py             ← working Gradio demo
├── api/                           ← NEW
│   ├── __init__.py
│   ├── server.py                  ← FastAPI app
│   ├── schemas.py                 ← Pydantic request/response models
│   └── inference.py               ← model loading + inference
├── infrastructure/
│   └── docker/
│       └── Dockerfile             ← NEW
├── notebooks/
│   └── training_curves.ipynb      ← NEW
├── data/
├── checkpoints/                   ← download from Google Drive (not in git)
├── weights/                       ← HuggingFace cache (not in git)
├── requirements.txt
└── README.md                      ← needs enterprise rewrite
```

---

## References

- LLaVA-1.5: https://arxiv.org/abs/2310.03744
- u-LLaVA: https://arxiv.org/abs/2311.05348
- PixelLLM: https://arxiv.org/abs/2312.09237
- GIoU: https://arxiv.org/abs/1902.09630
- RefCOCO: https://arxiv.org/abs/1608.00272