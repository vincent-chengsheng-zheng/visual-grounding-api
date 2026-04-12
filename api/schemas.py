"""
api/schemas.py

Pydantic request/response models for the visual grounding API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    text: str = Field(..., description="Referring expression, e.g. 'the red chair'")
    model: str = Field(
        default="main",
        description="Model to use: 'baseline', 'ablation', or 'main'",
    )


class BBoxResponse(BaseModel):
    bbox: List[float] = Field(..., description="Predicted bbox [xc, yc, w, h] in (0, 1)")
    model: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    device: str


class ModelInfo(BaseModel):
    name: str
    description: str
    test_iou: float
    test_rmse: float
    test_mae: float


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
