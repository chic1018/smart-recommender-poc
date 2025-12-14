from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

DEFAULT_TIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M")


def _parse_time(value: Union[str, int, float, datetime]) -> datetime:
    # ... (Implementation omitted)
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(int(value))
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1]
        for fmt in DEFAULT_TIME_FORMATS:
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass  # Fall through to ValueError
    raise ValueError(f"Unsupported time string: {value}")


def discretize_time(value: Union[str, int, float, datetime], bucket_minutes: int) -> Tuple[int, bool]:
    # ... (Implementation omitted)
    if bucket_minutes <= 0 or (1440 % bucket_minutes) != 0:
        raise ValueError("bucket_minutes must divide 1440 evenly")
    dt = _parse_time(value)
    total_minutes = dt.hour * 60 + dt.minute
    time_bucket = total_minutes // bucket_minutes
    is_weekend = dt.weekday() >= 5
    return int(time_bucket), bool(is_weekend)


def bucket_temperature(temp: Optional[float]) -> int:
    # ... (Implementation omitted)
    if temp is None: return 1
    v = float(temp)
    if v < 17.0: return 0
    if v < 23.0: return 1
    return 2


def bucket_light(light: Optional[float]) -> int:
    # ... (Implementation omitted)
    if light is None: return 2
    v = float(light)
    if v < 50.0: return 0
    if v < 300.0: return 1
    return 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# -------------------------
# Feature & Vocabulary Mapping
# -------------------------
# We define a global vocabulary of all possible features (Feature Indexing)

VOCAB_MAP = {}
VOCAB_SIZE = 0


def _map_feature(key: str, value: Any) -> str:
    """Creates a unique string identifier for a feature value."""
    return f"{key}={value}"


def _build_vocabulary(history: Iterable[Dict[str, Any]], bucket_minutes: int = 60) -> Tuple[Dict[str, int], List[str]]:
    """Generates the global feature vocabulary and maps them to indices."""
    global VOCAB_MAP, VOCAB_SIZE

    temp_vocab = set()
    device_vocab = set()

    for ev in history:
        try:
            # 1. Device and Action
            device = str(ev["device_id"])
            action = "ON" if str(ev.get("action", "")).upper() in {"ON", "OPEN", "START"} else "OFF"
            device_vocab.add(device)

            # 2. Time Features
            tb, iw = discretize_time(ev["event_time"], bucket_minutes)
            temp_b = bucket_temperature(ev.get("temperature"))
            light_b = bucket_light(ev.get("light_intensity"))

            temp_vocab.add(_map_feature("time_bucket", tb))
            temp_vocab.add(_map_feature("is_weekend", iw))
            temp_vocab.add(_map_feature("temp_bucket", temp_b))
            temp_vocab.add(_map_feature("light_bucket", light_b))

            # Optional: Feature Crossing (Time x Env)
            temp_vocab.add(_map_feature("time_x_temp", f"{tb}_{temp_b}"))
            temp_vocab.add(_map_feature("time_x_light", f"{tb}_{light_b}"))

        except Exception:
            continue

    # Index the vocabulary
    # Feature indices
    feature_list = sorted(list(temp_vocab))
    VOCAB_MAP = {f: i for i, f in enumerate(feature_list)}

    # Device indices
    device_list = sorted(list(device_vocab))
    VOCAB_MAP.update({d: i + len(feature_list) for i, d in enumerate(device_list)})

    VOCAB_SIZE = len(VOCAB_MAP)

    return VOCAB_MAP, device_list


# -------------------------
# Contextual Linear Model (CLM)
# -------------------------
class ContextualLinearModel:
    """
    v3 Contextual Linear Model (CLM) based on Logistic Regression / Wide Model.

    It predicts P(Device ON | Context) using linear combinations of feature cross weights.
    """

    def __init__(
            self,
            bucket_minutes: int = 60,
            device_on_weight: float = 1.0,  # Weight for positive samples during mock training
    ):
        self.bucket_minutes = int(bucket_minutes)
        self.device_on_weight = device_on_weight

        # Global Vocab and Model State
        self._vocab_map: Dict[str, int] = {}
        self._device_list: List[str] = []
        self._feature_dim: int = 0
        self._num_devices: int = 0

        # Model Weights (Initialized in build_from_history)
        # W: (Feature_Dim x Num_Devices) - Weight matrix for P(Device_i ON | Features)
        self.W: Optional[np.ndarray] = None
        # B: (Num_Devices) - Bias vector
        self.B: Optional[np.ndarray] = None

    # -------------------------
    # Core Feature Vectorization (phi(X))
    # -------------------------
    def _create_feature_vector(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Generates a sparse feature vector (one-hot encoded) from the current context.
        """
        if self._feature_dim == 0:
            return np.zeros(0)

        feature_vector = np.zeros(self._feature_dim)

        # 1. Extract context features
        try:
            tb, iw = discretize_time(context["event_time"], self.bucket_minutes)
            temp_b = bucket_temperature(context.get("temperature"))
            light_b = bucket_light(context.get("light_intensity"))
        except KeyError:
            # Must have event_time for context
            return feature_vector

        # 2. Add features to vector (using global vocabulary index)
        features = [
            _map_feature("time_bucket", tb),
            _map_feature("is_weekend", iw),
            _map_feature("temp_bucket", temp_b),
            _map_feature("light_bucket", light_b),
            _map_feature("time_x_temp", f"{tb}_{temp_b}"),
            _map_feature("time_x_light", f"{tb}_{light_b}"),
        ]

        for f in features:
            idx = self._vocab_map.get(f)
            if idx is not None and idx < self._feature_dim:
                feature_vector[idx] = 1.0

        return feature_vector

    # -------------------------
    # Build (Mock Training/Weight Loading)
    # -------------------------
    def build_from_history(self, history: Iterable[Dict[str, Any]]) -> None:
        """
        Builds the feature vocabulary and performs a mock training (or loads weights).
        """
        history_list = list(history)
        if not history_list:
            return

        # 1. Build Vocabulary
        vocab_map, device_list = _build_vocabulary(history_list, self.bucket_minutes)
        self._vocab_map = vocab_map
        self._device_list = device_list
        self._num_devices = len(device_list)

        # Feature dimensions are only the contextual features, not the devices themselves
        self._feature_dim = max(i for f, i in vocab_map.items() if f not in device_list) + 1

        if self._feature_dim == 0 or self._num_devices == 0:
            print("ERROR: Not enough data to build vocabulary.")
            return

        # 2. Mock Weight Initialization (Simulating a load from pre-trained model)
        # In a real scenario, W and B would be loaded from a file trained via L-BFGS or SGD
        np.random.seed(42)  # For reproducibility
        scale = 1.0 / math.sqrt(self._feature_dim + self._num_devices)
        self.W = np.random.uniform(-scale, scale, (self._feature_dim, self._num_devices))
        self.B = np.zeros(self._num_devices)

        print(f"INFO: CLM Model initialized. Features: {self._feature_dim}, Devices: {self._num_devices}")

        # OPTIONAL: Mock Weight Adjustment based on history (Simulating online fine-tuning/Learning)
        # This is the simplified version of the old frequency model, but mapped to linear weights.
        # We can simulate strong weight adjustment for high-frequency events.

        freq_map = defaultdict(lambda: defaultdict(int))  # (Feature) -> Device -> Count
        for ev in history_list:
            try:
                device = str(ev["device_id"])
                is_on = str(ev.get("action", "")).upper() in {"ON", "OPEN", "START"}

                # Get context features
                tb, iw = discretize_time(ev["event_time"], self.bucket_minutes)
                temp_b = bucket_temperature(ev.get("temperature"))
                light_b = bucket_light(ev.get("light_intensity"))

                features = [
                    _map_feature("time_bucket", tb),
                    _map_feature("is_weekend", iw),
                    _map_feature("temp_bucket", temp_b),
                    _map_feature("light_bucket", light_b),
                    _map_feature("time_x_temp", f"{tb}_{temp_b}"),
                    _map_feature("time_x_light", f"{tb}_{light_b}"),
                ]

                if device in self._device_list and is_on:
                    d_idx = self._device_list.index(device)
                    self.B[d_idx] += 0.01  # Global device preference adjustment

                    for f in features:
                        f_idx = self._vocab_map.get(f)
                        if f_idx is not None and f_idx < self._feature_dim:
                            # Reinforce positive feature-device connection
                            self.W[f_idx, d_idx] += 0.05 * self.device_on_weight
            except Exception:
                continue

    # -------------------------
    # Recommend
    # -------------------------
    def recommend(self, context: Union[Dict[str, Any], Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predicts P(Device ON | Context) using the linear model (W * X + B).
        """
        if self.W is None or self.B is None or self._feature_dim == 0:
            return []

        # 1. Vectorize Context
        if hasattr(context, "model_dump"):
            context = context.model_dump()

        # context must be dict-like, but history is ignored in this model
        if not isinstance(context, dict):
            raise TypeError("context must be a dict-like object")

        feature_vector_x = self._create_feature_vector(context)  # (1 x Feature_Dim)

        if feature_vector_x.size != self._feature_dim:
            # Should not happen if history builds correctly
            return []

        # 2. Linear Prediction (Dot Product)
        # Logits = X * W + B
        # (1 x F) @ (F x D) + (1 x D) -> (1 x D)
        logits = feature_vector_x @ self.W + self.B

        # 3. Activation (Logistic Regression / Sigmoid)
        probabilities = sigmoid(logits)

        # 4. Ranking and Formatting
        results: List[Dict[str, Any]] = []

        # Get sorted indices (descending probability)
        sorted_indices = np.argsort(probabilities)[::-1]

        for rank, d_idx in enumerate(sorted_indices):
            device_id = self._device_list[d_idx]
            score = probabilities[d_idx]

            # Use weights for simple explanation
            feature_weights = self.W[:, d_idx] * feature_vector_x
            top_features_indices = np.argsort(feature_weights)[::-1]

            top_features = []

            # Look up feature names from index
            idx_to_feature = {i: f for f, i in self._vocab_map.items() if i < self._feature_dim}

            for i in top_features_indices[:3]:
                if feature_vector_x[i] > 0 and feature_weights[i] > 0.01:
                    top_features.append(f"{idx_to_feature.get(i, 'Unknown')} ({feature_weights[i]:.4f})")

            results.append({
                "device_id": device_id,
                "score": round(float(score), 6),
                "details": {
                    "explanation_model": "Contextual Linear Model",
                    "bias_score": round(float(self.B[d_idx]), 4),
                    "context_features_contribution": ", ".join(top_features) or "None",
                    "rank": rank + 1
                }
            })

        return results[: max(0, int(top_k))]

    # -------------------------
    # Utilities
    # -------------------------
    def export_model(self) -> Dict[str, Any]:
        """Exports the model state (weights) to a serializable dictionary."""
        # Convert NumPy arrays to list for JSON serialization
        return {
            "model_type": "ContextualLinearModel",
            "W": self.W.tolist() if self.W is not None else [],
            "B": self.B.tolist() if self.B is not None else [],
            "vocab_map": self._vocab_map,
            "device_list": self._device_list,
            "meta": {
                "feature_dim": self._feature_dim,
                "num_devices": self._num_devices,
                "bucket_minutes": self.bucket_minutes,
            }
        }

    def devices(self) -> List[str]:
        return self._device_list
