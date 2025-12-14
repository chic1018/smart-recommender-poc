"""
Time discretization utilities.

Provides:
- parsing of common time inputs (ISO-like string, datetime, epoch seconds)
- conversion to configurable time buckets (hourly or half-hourly or custom minutes)
- weekend check and simple time-of-day helpers

Functions:
- parse_time(value) -> datetime.datetime
- get_time_bucket(dt, bucket_minutes=60) -> int
- is_weekend(dt) -> bool
- extract_time_features(value, bucket_minutes=60) -> dict
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Union, Dict

DEFAULT_TIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M")


def parse_time(value: Union[str, int, float, datetime]) -> datetime:
    """
    Parse a timestamp value into a timezone-naive datetime (local time assumed).
    Accepts:
      - datetime -> returned (if tz-aware, converted to naive local via .astimezone())
      - integer/float -> treated as Unix epoch seconds
      - string -> tried against common patterns (raises ValueError if none match)

    Note: For POC we treat parsed datetimes as local time (no timezone conversion).
    """
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(int(value))
    if isinstance(value, str):
        for fmt in DEFAULT_TIME_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        # last resort: try ISO parse via fromisoformat
        try:
            return datetime.fromisoformat(value)
        except Exception as e:
            raise ValueError(f"Unsupported time string format: {value}") from e
    raise TypeError("Unsupported type for parse_time; expected str/int/float/datetime")


def get_time_bucket(dt: datetime, bucket_minutes: int = 60) -> int:
    """
    Map a datetime to a bucket index.
    - bucket_minutes: size of each bucket in minutes. Typical values: 60 (hourly), 30 (half-hour).
    Returns:
      integer bucket index in range [0, 24*60/bucket_minutes - 1]
    """
    if bucket_minutes <= 0 or (60 * 24) % bucket_minutes != 0:
        raise ValueError("bucket_minutes must divide 1440 (minutes per day) evenly and be > 0")
    total_minutes = dt.hour * 60 + dt.minute
    return total_minutes // bucket_minutes


def is_weekend(dt: datetime) -> bool:
    """Return True if dt falls on a weekend (Saturday=5 or Sunday=6)."""
    return dt.weekday() >= 5


def extract_time_features(value: Union[str, int, float, datetime],
                          bucket_minutes: int = 60) -> Dict[str, Union[int, bool]]:
    """
    Convenience: parse value and return a small feature dict:
      {
        "time_bucket": int,
        "is_weekend": bool,
        "hour": int,
        "minute": int
      }
    """
    dt = parse_time(value)
    return {
        "time_bucket": get_time_bucket(dt, bucket_minutes=bucket_minutes),
        "is_weekend": is_weekend(dt),
        "hour": dt.hour,
        "minute": dt.minute
    }


# Example (commented):
# >>> extract_time_features("2025-12-01 06:52", bucket_minutes=60)
# {'time_bucket': 6, 'is_weekend': False, 'hour': 6, 'minute': 52}
