from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.contextual_linear.model import ContextualLinearModel

router = APIRouter(
    prefix="/recommend",
    tags=["recommendation"],
)


# =========================================================
# Request / Response Schemas
# =========================================================

class RecommendContext(BaseModel):
    event_time: str
    temperature: float | None = None
    humidity: float | None = None
    light_intensity: float | None = None


class RecommendRequest(BaseModel):
    history: List[Dict[str, Any]] = Field(..., description="User interaction history")
    context: RecommendContext
    top_k: int = 3


class RecommendResponse(BaseModel):
    model_name: str
    recommendations: List[Dict[str, Any]]
    meta: Dict[str, Any]


# =========================================================
# Endpoint
# =========================================================

@router.post("/contextual-frequency", response_model=RecommendResponse)
def recommend_contextual_frequency(payload: RecommendRequest):
    if not payload.history:
        raise HTTPException(status_code=400, detail="History is empty")

    model = ContextualLinearModel(
        bucket_minutes=60,
    )

    model.build_from_history(payload.history)

    recommendations = model.recommend(
        context=payload.context.model_dump(),
        top_k=payload.top_k,
    )

    return {
        "model_name": "ContextualFrequency",
        "recommendations": recommendations,
        "meta": {
            "bucket_minutes": model.bucket_minutes,
        },
    }
