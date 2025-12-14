"""
Humidity discretization.

Design:
- For extreme-low-resource usage we discretize humidity into a small number of buckets.
- Defaults chosen for general indoor comfort zones:
    0: low (dry)      -> humidity <= 30%
    1: normal         -> 30% < humidity <= 60%
    2: high (humid)   -> humidity > 60%

API:
- bucket_humidity(value, thresholds=None) -> int
- humidity_label(bucket, labels=None) -> str
"""

from typing import Iterable, Tuple, List

DEFAULT_HUMIDITY_THRESHOLDS: Tuple[float, float] = (30.0, 60.0)
DEFAULT_HUMIDITY_LABELS: Tuple[str, str, str] = ("low", "normal", "high")


def _validate_value(value: float):
    if value is None:
        raise ValueError("humidity value is None")
    try:
        v = float(value)
    except Exception:
        raise TypeError("humidity value must be numeric")
    if v < 0 or v > 100:
        raise ValueError("humidity should be in [0, 100]")
    return v


def bucket_humidity(value: float,
                    thresholds: Iterable[float] = DEFAULT_HUMIDITY_THRESHOLDS) -> int:
    """
    Map humidity (0..100) to bucket index (0..len(thresholds)).
    thresholds must be increasing and define the upper bound of each lower bucket.
    Example with defaults (30, 60):
      value <=30 -> bucket 0
      30 < value <=60 -> bucket 1
      value > 60 -> bucket 2
    """
    v = _validate_value(value)
    th = list(thresholds)
    if any(t < 0 or t > 100 for t in th):
        raise ValueError("thresholds must be in [0, 100]")
    if any(th[i] >= th[i + 1] for i in range(len(th) - 1)):
        raise ValueError("thresholds must be strictly increasing")
    # find first threshold >= v
    for i, bound in enumerate(th):
        if v <= bound:
            return i
    return len(th)


def humidity_label(bucket: int,
                   labels: Iterable[str] = DEFAULT_HUMIDITY_LABELS) -> str:
    """Return human-friendly label for bucket index."""
    lbls = list(labels)
    if bucket < 0 or bucket >= len(lbls):
        raise IndexError("bucket out of range for labels")
    return lbls[bucket]


# Example:
# >>> bucket_humidity(62.0) -> 2
# >>> humidity_label(2) -> "high"
