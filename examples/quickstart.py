"""Minimal synthetic smoke test for DFAA-MemoryNet."""

import numpy as np

from dfaa_memorynet import (
    FULL_FEATURE_NAMES,
    build_memorynet,
)

PATCH_SIZE = 32
CHANNELS = len(FULL_FEATURE_NAMES)

model = build_memorynet(
    input_shape=(PATCH_SIZE, PATCH_SIZE, CHANNELS),
    base_filters=24,
    dropout_scale=1.05,
)

x = np.zeros(
    (1, PATCH_SIZE, PATCH_SIZE, CHANNELS),
    dtype=np.float32,
)

y = model.predict(x, verbose=0)

print("DFAA-MemoryNet quick test")
print("Input shape :", x.shape)
print("Output shape:", y.shape)
print("Fraction channel range:", float(y[..., 0].min()), float(y[..., 0].max()))
print("Event channel range   :", float(y[..., 1].min()), float(y[..., 1].max()))

assert y.shape == (1, PATCH_SIZE, PATCH_SIZE, 2)
print("OK")
