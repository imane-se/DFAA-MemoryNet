"""Predictor construction utilities for DFAA-MemoryNet."""

from __future__ import annotations

from collections import OrderedDict
import numpy as np
import pandas as pd
import xarray as xr

FULL_FEATURE_NAMES = [
    "pe_t",
    "dsm_t",
    "pe_mean2",
    "pe_mean4",
    "dsm_mean2",
    "dsm_mean4",
    "pe_slope1",
    "dsm_slope1",
    "memory_persistence_t",
    "memory_decay_speed_t",
    "memory_instability_t",
    "memory_retention_t",
    "pe_tplus1",
    "pe_tplus2",
    "pe_tplus3",
    "pe_tplus4",
    "persistence_static",
    "decay_speed_static",
    "instability_static",
    "memory_retention_static",
    "memory_class",
    "land_organization",
    "x_norm",
    "y_norm",
    "valid_mask",
]

SMM_FEATURES = [
    "memory_persistence_t",
    "memory_decay_speed_t",
    "memory_instability_t",
    "memory_retention_t",
    "persistence_static",
    "decay_speed_static",
    "instability_static",
    "memory_retention_static",
    "memory_class",
]


def recent_mean(da: xr.DataArray, window: int) -> xr.DataArray:
    """Dask-safe trailing mean including the current week."""
    if window < 1:
        raise ValueError("window must be >= 1")

    shifted = [da.shift(time=k) for k in range(window)]
    stacked = xr.concat(
        shifted,
        dim=pd.Index(range(window), name="_recent_lag"),
    )
    return stacked.mean("_recent_lag", skipna=True).astype("float32")


def recent_slope(da: xr.DataArray, lag: int = 1) -> xr.DataArray:
    """Current-minus-lagged predictor tendency."""
    if lag < 1:
        raise ValueError("lag must be >= 1")
    return (da - da.shift(time=lag)).astype("float32")


def normalized_coordinates(
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return x/y coordinates normalized to [-1, 1]."""
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lon, lat)

    x = (
        2 * (lon2d - np.nanmin(lon2d))
        / (np.nanmax(lon2d) - np.nanmin(lon2d) + 1e-6)
        - 1
    )
    y = (
        2 * (lat2d - np.nanmin(lat2d))
        / (np.nanmax(lat2d) - np.nanmin(lat2d) + 1e-6)
        - 1
    )

    coords = {"lat": lat, "lon": lon}

    return (
        xr.DataArray(x.astype("float32"), dims=("lat", "lon"), coords=coords, name="x_norm"),
        xr.DataArray(y.astype("float32"), dims=("lat", "lon"), coords=coords, name="y_norm"),
    )


def build_predictor_dict(
    z_pe: xr.DataArray,
    z_dsm: xr.DataArray,
    memory_persistence_t: xr.DataArray,
    memory_decay_speed_t: xr.DataArray,
    memory_instability_t: xr.DataArray,
    memory_retention_t: xr.DataArray,
    pe_future: xr.DataArray,
    persistence_static: xr.DataArray,
    decay_speed_static: xr.DataArray,
    instability_static: xr.DataArray,
    memory_retention_static: xr.DataArray,
    memory_class: xr.DataArray,
    land_organization: xr.DataArray,
    valid_mask: xr.DataArray,
    horizon: int = 4,
) -> OrderedDict:
    """Construct predictors in the exact reference feature order.

    `pe_future` must have dimensions `(lead, time, lat, lon)` or equivalent
    coordinates with leads 1..horizon.
    """
    x_norm, y_norm = normalized_coordinates(z_pe.lat.values, z_pe.lon.values)

    features = OrderedDict([
        ("pe_t", z_pe),
        ("dsm_t", z_dsm),
        ("pe_mean2", recent_mean(z_pe, 2)),
        ("pe_mean4", recent_mean(z_pe, 4)),
        ("dsm_mean2", recent_mean(z_dsm, 2)),
        ("dsm_mean4", recent_mean(z_dsm, 4)),
        ("pe_slope1", recent_slope(z_pe, 1)),
        ("dsm_slope1", recent_slope(z_dsm, 1)),
        ("memory_persistence_t", memory_persistence_t),
        ("memory_decay_speed_t", memory_decay_speed_t),
        ("memory_instability_t", memory_instability_t),
        ("memory_retention_t", memory_retention_t),
    ])

    for lead in range(1, horizon + 1):
        features[f"pe_tplus{lead}"] = pe_future.sel(lead=lead)

    features.update(OrderedDict([
        ("persistence_static", persistence_static),
        ("decay_speed_static", decay_speed_static),
        ("instability_static", instability_static),
        ("memory_retention_static", memory_retention_static),
        ("memory_class", memory_class.astype("float32")),
        ("land_organization", land_organization.astype("float32")),
        ("x_norm", x_norm),
        ("y_norm", y_norm),
        ("valid_mask", valid_mask.astype("float32")),
    ]))

    if list(features) != FULL_FEATURE_NAMES:
        raise RuntimeError("Reference feature order was not preserved.")

    return features


def feature_names_for_variant(variant: str) -> list[str]:
    """Return feature names for the four model configurations."""
    variant = variant.lower().strip()

    if variant in {"full", "full_memorynet", "dfaa-memorynet"}:
        return FULL_FEATURE_NAMES.copy()

    if variant in {"without_smm", "without_memory", "nomemory"}:
        return [f for f in FULL_FEATURE_NAMES if f not in SMM_FEATURES]

    if variant in {"forcing_current_rt1", "forcing+current_rt1"}:
        keep = [
            "pe_t", "dsm_t",
            "pe_mean2", "pe_mean4",
            "dsm_mean2", "dsm_mean4",
            "pe_slope1", "dsm_slope1",
            "pe_tplus1", "pe_tplus2", "pe_tplus3", "pe_tplus4",
            "x_norm", "y_norm", "valid_mask",
        ]
        return keep

    if variant in {"forcing_only", "forcing"}:
        return [
            "pe_t",
            "pe_mean2",
            "pe_mean4",
            "pe_slope1",
            "pe_tplus1",
            "pe_tplus2",
            "pe_tplus3",
            "pe_tplus4",
            "x_norm",
            "y_norm",
            "valid_mask",
        ]

    raise ValueError(f"Unknown model variant: {variant}")


def fit_normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel mean/std from training data only."""
    mean = np.nanmean(x, axis=(0, 1, 2), keepdims=True).astype(np.float32)
    std = np.nanstd(x, axis=(0, 1, 2), keepdims=True).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def apply_normalizer(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply training-only normalization and replace non-finite values."""
    return np.nan_to_num(
        (x - mean) / std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
