"""DFAA-MemoryNet software package."""

__version__ = "1.0.0"

from .memorynet import build_memorynet, make_dualtask_loss
from .predictors import (
    FULL_FEATURE_NAMES,
    SMM_FEATURES,
    feature_names_for_variant,
    fit_normalizer,
    apply_normalizer,
)
from .inference import predict_map

__all__ = [
    "build_memorynet",
    "make_dualtask_loss",
    "FULL_FEATURE_NAMES",
    "SMM_FEATURES",
    "feature_names_for_variant",
    "fit_normalizer",
    "apply_normalizer",
    "predict_map",
]
