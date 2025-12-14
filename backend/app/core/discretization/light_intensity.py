"""
Light intensity discretization.

Default thresholds (lux-like numbers) tuned for indoor use:
  0..50        -> dark
  51..300      -> dim / indoor morning
  301..1000+   -> bright / daylight

API:
- bucket_light(value, thresholds=None) -> int
- light_label(bucket, labels=None) -> str

Notes:
- light intensity sensor units vary by hardware. Thresholds here are approximate and configurable.
- Values <= 0 are treated as 0 (very dark).
"""

from typing import Iterable, List

DEFAULT_LIGHT_THRESHOLDS = (50.0, 300.0)
DEFAULT_LIGHT_LABELS = ("dark", "dim", "bright")


def _validate_value(value: float):
    if value is None:
        raise ValueError("light intensity is None")
    try:
        v = float(value)
    except Exception:
        raise TypeError("light intensity must be numeric")
    # negative light is invalid, clamp or reject - here we reject
    if v < 0:
        raise ValueError("light intensity must be >= 0")
    return v


def bucket_light(value: float,
                 thresholds: Iterable[float] = DEFAULT_LIGHT_THRESHOLDS) -> int:
    """
    Map light intensity to a small discrete bucket index.
    thresholds must be increasing:
      value <= thresholds[0] -> bucket 0 (dark)
      thresholds[0] < value <= thresholds[1] -> bucket 1 (dim)
      value > thresholds[1] -> bucket 2 (bright)
    """
    v = _validate_value(value)
    th = list(thresholds)
    if any(t < 0 for t in th):
        raise ValueError("thresholds must be >= 0")
    if any(th[i] >= th[i + 1] for i in range(len(th) - 1)):
        raise ValueError("thresholds must be strictly increasing")
    for i, bound in enumerate(th):
        if v <= bound:
            return i
    return len(th)


def light_label(bucket: int,
                labels: Iterable[str] = DEFAULT_LIGHT_LABELS) -> str:
    lbls = list(labels)
    if bucket < 0 or bucket >= len(lbls):
        raise IndexError("bucket out of range for labels")
    return lbls[bucket]


# Example:
# >>> bucket_light(30) -> 0  # dark
# >>> bucket_light(120) -> 1 # dim
# >>> bucket_light(400) -> 2 # bright
