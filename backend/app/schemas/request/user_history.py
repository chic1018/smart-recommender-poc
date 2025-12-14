from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class EventItem(BaseModel):
    """
    Single event record as passed from frontend.
    event_time may be ISO string or unix epoch (int), or datetime.
    action: "ON" / "OFF" (case-insensitive accepted)
    sensor fields optional (None if missing).
    """
    event_time: Union[str, int, float, datetime] = Field(..., description="Timestamp (ISO string or epoch seconds)")
    device_id: str = Field(..., description="Device identifier string")
    action: str = Field(..., description='Action, expected "ON" or "OFF"')
    temperature: Optional[float] = Field(None, description="Temperature in Celsius (optional)")
    humidity: Optional[float] = Field(None, description="Relative humidity in % (optional)")
    light_intensity: Optional[float] = Field(None, description="Light intensity (lux-like) (optional)")

    @validator("action")
    def action_must_be_on_off(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("action must be a string")
        uv = v.strip().upper()
        if uv not in {"ON", "OFF", "1", "0"}:
            raise ValueError('action must be "ON" or "OFF" (or equivalent "1"/"0")')
        return uv


class CurrentContext(BaseModel):
    """
    Current context for which recommendation is requested.
    Only raw observation fields are accepted for the POC:
      - event_time (ISO string / epoch / datetime)
      - temperature (float; optional)
      - humidity (float; optional)
      - light_intensity (float; optional)

    The backend will perform time parsing and discretization (time bucket,
    weekend flag) and bucket mapping for temperature/light.
    """
    event_time: Union[str, int, float, datetime] = Field(..., description="Current timestamp (ISO string or epoch seconds)")
    temperature: Optional[float] = Field(None, description="Current temperature (optional)")
    humidity: Optional[float] = Field(None, description="Current humidity (optional)")
    light_intensity: Optional[float] = Field(None, description="Current light intensity (optional)")

    @validator("event_time")
    def event_time_must_be_present(cls, v):
        if v is None:
            raise ValueError("event_time must be provided in current_context")
        return v


class ModelConfig(BaseModel):
    """
    Config shared by both endpoints. Frontend supplies desired modeling hyper-params.
    Note: NaiveBayes endpoint will read only the fields it needs (e.g. smoothing).
    """
    bucket_minutes: int = Field(60, description="Time bucket size in minutes (must divide 1440)")
    min_support: int = Field(1, description="Minimum support used by contextual-frequency fallback logic")
    beta: float = Field(1.0, description="Weight for time-only counts when blending (contextual-frequency)")
    gamma: float = Field(0.0, description="Weight for global counts when blending (contextual-frequency)")
    blend: bool = Field(True, description="Whether to blend levels (contextual-frequency)")
    top_k: int = Field(3, description="Number of top results requested")
    smoothing: float = Field(1.0, description="Smoothing parameter for Naive Bayes (Laplace).")

    @validator("bucket_minutes")
    def bucket_must_divide_day(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("bucket_minutes must be positive")
        if 1440 % v != 0:
            raise ValueError("bucket_minutes must divide 1440 (minutes per day) evenly")
        return v

    @validator("top_k")
    def top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v

    @validator("min_support")
    def min_support_nonnegative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("min_support must be >= 0")
        return v


class UserHistoryRequest(BaseModel):
    """
    Request body accepted by both /recommend/contextual-frequency and /recommend/naive-bayes.
    - history: list of historical events (frontend-controlled)
    - current_context: the current observed context (raw fields only)
    - config: model configuration / hyper-params (frontend-controlled)
    """
    history: List[EventItem] = Field(..., description="User history: list of events (most recent last or arbitrary order)")
    current_context: CurrentContext = Field(..., description="Current context (raw observations only)")
    config: ModelConfig = Field(default_factory=ModelConfig, description="Modeling & inference configuration")

    @validator("history")
    def history_not_empty(cls, v: List[EventItem]) -> List[EventItem]:
        if not v or len(v) == 0:
            raise ValueError("history must contain at least one event")
        return v
