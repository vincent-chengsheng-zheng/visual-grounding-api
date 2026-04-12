"""
api/inference.py

Model loading and inference functions for the FastAPI server.

Two loading strategies controlled by LOAD_MODE env var:
  - "sequential" (default): One model in VRAM at a time (~14GB).
    Swaps models on demand. Works on any GPU with >= 16GB VRAM.
    First request after a switch takes ~30-60s.
  - "parallel": All models loaded simultaneously (~42GB).
    Instant switching. Requires >= 48GB VRAM (e.g. A100 80GB).

Usage:
    LOAD_MODE=sequential uvicorn api.server:app ...   # RTX 5090 (32GB)
    LOAD_MODE=parallel  uvicorn api.server:app ...   # A100 (80GB)
"""

import gc
import os
import sys
import time
import base64
import io
from typing import Dict

import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data.preprocessing import PROMPT_TEMPLATE
from core.paths import PATHS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODE = os.environ.get("LOAD_MODE", "sequential")

CKPT_ABLATION = str(PATHS.ckpt_ablation / "best.pth")
CKPT_MAIN = str(PATHS.ckpt_main / "best.pth")


# ── Base class ───────────────────────────────────────────────────────────────

class BaseRegistry:
    def __init__(self):
        self.loaded_models: list[str] = []
        self.active_model: str = ""

    def load_all(self):
        raise NotImplementedError

    def predict(self, image_b64: str, text: str, model_name: str) -> Dict:
        raise NotImplementedError


# ── Sequential: one model in VRAM at a time ──────────────────────────────────

class SequentialRegistry(BaseRegistry):
    """
    Loads one model at a time. Unloads the current model before loading
    the next. Uses ~14GB VRAM regardless of how many models are available.
    """

    def __init__(self):
        super().__init__()
        self._model = None
        self._processor = None

    def load_all(self):
        print(f"\n[API] Mode: SEQUENTIAL (one model in VRAM at a time)")
        print(f"[API] Device: {DEVICE}")

        self.loaded_models = ["baseline"]
        if os.path.exists(CKPT_ABLATION):
            self.loaded_models.append("ablation")
        if os.path.exists(CKPT_MAIN):
            self.loaded_models.append("main")

        print(f"[API] Available models: {self.loaded_models}")

        # Pre-load main by default
        default = "main" if "main" in self.loaded_models else self.loaded_models[0]
        self._load_model(default)
        print(f"[API] Ready. Active: {self.active_model}\n")

    def _unload(self):
        if self._model is not None:
            print(f"[API] Unloading {self.active_model}...")
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self.active_model = ""
            gc.collect()
            torch.cuda.empty_cache()

    def _load_model(self, model_name: str):
        if model_name == self.active_model:
            return
        self._unload()

        t0 = time.time()
        print(f"[API] Loading {model_name}...")

        if model_name == "baseline":
            from core.model.llava import StandardLLaVA
            self._model = StandardLLaVA(
                model_id="llava-hf/llava-1.5-7b-hf",
                hf_cache=str(PATHS.weights),
            )
            self._model.to(DEVICE).eval()
            self._processor = self._model.processor

        elif model_name in ("ablation", "main"):
            from core.model.spatial_llava import load_model
            use_lora = (model_name == "main")
            self._model, self._processor = load_model(
                use_lora=use_lora, device=DEVICE
            )
            ckpt_path = CKPT_MAIN if model_name == "main" else CKPT_ABLATION
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            self._model.load_state_dict(ckpt["model_state"], strict=False)
            self._model.eval()

        self.active_model = model_name
        print(f"[API] {model_name} loaded in {time.time() - t0:.1f}s")

    def predict(self, image_b64: str, text: str, model_name: str) -> Dict:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        self._load_model(model_name)

        if model_name == "baseline":
            return self._predict_baseline(img, text)
        else:
            return self._predict_spatial(img, text)

    def _predict_baseline(self, img: Image.Image, text: str) -> Dict:
        from core.model.llava import StandardLLaVA
        prompt = PROMPT_TEMPLATE.format(text=text)
        inputs = self._processor(
            text=[prompt], images=[img],
            return_tensors="pt", padding=True,
        ).to(DEVICE)

        t0 = time.time()
        with torch.no_grad():
            output_ids = self._model.model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
            )
        ms = (time.time() - t0) * 1000

        raw = self._processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        bbox = StandardLLaVA.parse_bbox(raw)
        return {
            "bbox": bbox if bbox else [0.0, 0.0, 0.0, 0.0],
            "inference_time_ms": round(ms, 1),
        }

    def _predict_spatial(self, img: Image.Image, text: str) -> Dict:
        prompt = PROMPT_TEMPLATE.format(text=text)
        inputs = self._processor(
            text=[prompt], images=[img],
            return_tensors="pt", padding=True,
        ).to(DEVICE)

        t0 = time.time()
        with torch.no_grad():
            pred = self._model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["pixel_values"],
            )
        ms = (time.time() - t0) * 1000

        return {
            "bbox": pred[0].cpu().tolist(),
            "inference_time_ms": round(ms, 1),
        }


# ── Parallel: all models loaded simultaneously ──────────────────────────────

class ParallelRegistry(BaseRegistry):
    """
    Loads all 3 models into VRAM at startup. Instant switching.
    Requires ~42GB VRAM (A100 80GB or similar).
    """

    def __init__(self):
        super().__init__()
        self._baseline = None
        self._ablation_model = None
        self._ablation_processor = None
        self._main_model = None
        self._main_processor = None

    def load_all(self):
        print(f"\n[API] Mode: PARALLEL (all models in VRAM)")
        print(f"[API] Device: {DEVICE}")

        # Baseline
        print("[API] 1/3 Baseline...")
        from core.model.llava import StandardLLaVA
        self._baseline = StandardLLaVA(
            model_id="llava-hf/llava-1.5-7b-hf",
            hf_cache=str(PATHS.weights),
        )
        self._baseline.to(DEVICE).eval()
        self.loaded_models.append("baseline")

        # Ablation
        if os.path.exists(CKPT_ABLATION):
            print("[API] 2/3 Ablation...")
            from core.model.spatial_llava import load_model
            self._ablation_model, self._ablation_processor = load_model(
                use_lora=False, device=DEVICE
            )
            ckpt = torch.load(CKPT_ABLATION, map_location=DEVICE, weights_only=False)
            self._ablation_model.load_state_dict(ckpt["model_state"], strict=False)
            self._ablation_model.eval()
            self.loaded_models.append("ablation")

        # Main
        if os.path.exists(CKPT_MAIN):
            print("[API] 3/3 Main (LoRA)...")
            from core.model.spatial_llava import load_model
            self._main_model, self._main_processor = load_model(
                use_lora=True, device=DEVICE
            )
            ckpt = torch.load(CKPT_MAIN, map_location=DEVICE, weights_only=False)
            self._main_model.load_state_dict(ckpt["model_state"], strict=False)
            self._main_model.eval()
            self.loaded_models.append("main")

        self.active_model = "all"
        print(f"[API] Ready. Loaded: {self.loaded_models}\n")

    def predict(self, image_b64: str, text: str, model_name: str) -> Dict:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        prompt = PROMPT_TEMPLATE.format(text=text)

        if model_name == "baseline":
            return self._predict_baseline(img, prompt)
        elif model_name == "ablation":
            return self._predict_spatial(img, prompt, self._ablation_model, self._ablation_processor)
        elif model_name == "main":
            return self._predict_spatial(img, prompt, self._main_model, self._main_processor)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _predict_baseline(self, img: Image.Image, prompt: str) -> Dict:
        from core.model.llava import StandardLLaVA
        inputs = self._baseline.processor(
            text=[prompt], images=[img],
            return_tensors="pt", padding=True,
        ).to(DEVICE)

        t0 = time.time()
        with torch.no_grad():
            output_ids = self._baseline.model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
            )
        ms = (time.time() - t0) * 1000

        raw = self._baseline.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        bbox = StandardLLaVA.parse_bbox(raw)
        return {
            "bbox": bbox if bbox else [0.0, 0.0, 0.0, 0.0],
            "inference_time_ms": round(ms, 1),
        }

    def _predict_spatial(self, img: Image.Image, prompt: str, model, processor) -> Dict:
        inputs = processor(
            text=[prompt], images=[img],
            return_tensors="pt", padding=True,
        ).to(DEVICE)

        t0 = time.time()
        with torch.no_grad():
            pred = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["pixel_values"],
            )
        ms = (time.time() - t0) * 1000

        return {
            "bbox": pred[0].cpu().tolist(),
            "inference_time_ms": round(ms, 1),
        }


# ── Pick strategy based on env var ───────────────────────────────────────────

if LOAD_MODE == "parallel":
    registry = ParallelRegistry()
else:
    registry = SequentialRegistry()
