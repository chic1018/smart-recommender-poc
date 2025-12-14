from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RecommendationCounts(BaseModel):
    """
    Optional counts breakdown returned by ContextualFrequency.
    Fields are optional because NaiveBayes may return likelihoods instead.
    """
    full: Optional[int] = Field(None, description="Count for full context (time+env)")
    time: Optional[int] = Field(None, description="Count for time-only (time_bucket+is_weekend)")
    global_count: Optional[int] = Field(None, description="Global count across history")


class RecommendationItem(BaseModel):
    """
    Single recommendation entry.
    - device_id: identifier
    - score: model-provided score (empirical count or log-prob or blended score)
    - counts: optional contextual counts for explainability
    - details: optional dict for model-specific diagnostics (e.g., log-probs per feature)
    """
    device_id: str = Field(..., description="Device identifier")
    score: float = Field(..., description="Score (higher is better); semantics depend on model")
    counts: Optional[RecommendationCounts] = Field(None, description="Optional counts-based breakdown")
    details: Optional[Dict[str, Any]] = Field(None, description="Model-specific details for explainability")


class RecommendationResponse(BaseModel):
    """
    Unified response model used by both endpoints.
    - model_name: string id of model used (e.g. "ContextualFrequency" or "NaiveBayes")
    - recommendations: ordered list of RecommendationItem (top_k)
    - meta: optional metadata (timing, event counts, hyperparams)
    """
    model_name: str = Field(..., description="Name of the model that produced the recommendations")
    recommendations: List[RecommendationItem] = Field(..., description="Top-K recommendations ordered by score desc")
    meta: Optional[Dict[str, Any]] = Field(None,
                                           description="Optional meta information (e.g. total_events, bucket_minutes)")
