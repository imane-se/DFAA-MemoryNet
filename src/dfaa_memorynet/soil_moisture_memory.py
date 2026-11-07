"""Generic helpers for preparing soil-moisture-memory predictor fields."""

from __future__ import annotations
import numpy as np
import xarray as xr


def memory_retention(
    persistence: xr.DataArray,
    decay_speed: xr.DataArray,
) -> xr.DataArray:
    """Combine persistence and decay into a bounded retention descriptor.

    Retention increases with persistence and decreases with decay speed.
    """
    return (
        persistence * (1.0 - decay_speed)
    ).clip(0, 1).astype("float32").rename("memory_retention")


def classify_memory_tertiles(
    retention_static: xr.DataArray,
) -> xr.DataArray:
    """Classify valid static retention into low/medium/high memory tertiles."""
    q = retention_static.quantile([1/3, 2/3], skipna=True)
    q1 = float(q.sel(quantile=1/3).values)
    q2 = float(q.sel(quantile=2/3).values)

    return xr.where(
        ~np.isfinite(retention_static),
        0,
        xr.where(
            retention_static <= q1,
            1,
            xr.where(retention_static <= q2, 2, 3),
        ),
    ).astype("int8").rename("memory_class")
