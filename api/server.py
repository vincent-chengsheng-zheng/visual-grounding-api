"""
api/server.py

FastAPI server for Spatial-LLaVA visual grounding.

Endpoints:
    POST /predict  — Run inference on a single image
    GET  /health   — Health check + loaded model status
    GET  /models   — List available models with test metrics

Run:
    cd visual-grounding-api
    export TRANSFORMERS_OFFLINE=1
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import (
    PredictRequest,
    BBoxResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
)
from api.inference import registry, DEVICE


MODEL_METRICS = {
    "baseline": {
        "description": "Standard LLaVA-1.5-7B + regex bbox parsing (no training)",
        "test_iou": 0.097,
        "test_rmse": 0.288,
        "test_mae": 0.238,
    },
    "ablation": {
        "description": "Frozen LLaVA backbone + MLP regression head",
        "test_iou": 0.284,
        "test_rmse": 0.224,
        "test_mae": 0.177,
    },
    "main": {
        "description": "LoRA fine-tuned LLaVA + MLP regression head",
        "test_iou": 0.386,
        "test_rmse": 0.172,
        "test_mae": 0.119,
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    t0 = time.time()
    registry.load_all()
    print(f"[API] All models loaded in {time.time() - t0:.1f}s")
    yield


app = FastAPI(
    title="Spatial-LLaVA Visual Grounding API",
    description=(
        "Visual grounding API that predicts bounding boxes from images and "
        "referring expressions using LLaVA-based models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=BBoxResponse)
async def predict(req: PredictRequest):
    """
    Predict a bounding box for a referring expression in an image.

    - **image**: Base64-encoded image (JPEG or PNG)
    - **text**: Referring expression, e.g. "the red chair"
    - **model**: One of "baseline", "ablation", "main" (default: "main")
    """
    if req.model not in registry.loaded_models:
        available = ", ".join(registry.loaded_models)
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' not available. Loaded: [{available}]",
        )

    try:
        result = registry.predict(req.image, req.text, req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return BBoxResponse(
        bbox=result["bbox"],
        model=req.model,
        inference_time_ms=result["inference_time_ms"],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns loaded models and device info."""
    return HealthResponse(
        status="ok",
        models_loaded=registry.loaded_models,
        device=DEVICE,
    )


@app.get("/models", response_model=ModelsResponse)
async def models():
    """List available models with their test-set metrics."""
    model_list = []
    for name in registry.loaded_models:
        metrics = MODEL_METRICS[name]
        model_list.append(
            ModelInfo(
                name=name,
                description=metrics["description"],
                test_iou=metrics["test_iou"],
                test_rmse=metrics["test_rmse"],
                test_mae=metrics["test_mae"],
            )
        )
    return ModelsResponse(models=model_list)
