"""Sliding-window inference for DFAA-MemoryNet."""

from __future__ import annotations
import numpy as np


def _origins(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_map(
    model,
    feature_cube: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    patch_size: int = 32,
    stride: int = 16,
    batch_size: int = 4,
) -> np.ndarray:
    """Predict a full spatial map by overlap averaging.

    Parameters
    ----------
    feature_cube
        Array `(height, width, channels)` in the exact training feature order.

    Returns
    -------
    np.ndarray
        `(height, width, 2)` with fraction and event-probability channels.
    """
    x = np.asarray(feature_cube, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("feature_cube must have shape (height, width, channels)")

    h, w, c = x.shape
    if h < patch_size or w < patch_size:
        raise ValueError("feature_cube is smaller than patch_size")

    mean_vec = np.asarray(mean).reshape(-1)
    std_vec = np.asarray(std).reshape(-1)

    if mean_vec.size != c or std_vec.size != c:
        raise ValueError("Normalization channel count does not match feature cube.")

    accum = np.zeros((h, w, 2), dtype=np.float64)
    count = np.zeros((h, w, 1), dtype=np.float64)

    patches = []
    locations = []

    def flush():
        if not patches:
            return
        batch = np.stack(patches).astype(np.float32)
        batch = np.nan_to_num(
            (batch - mean_vec.reshape(1, 1, 1, -1))
            / std_vec.reshape(1, 1, 1, -1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        pred = model.predict(batch, batch_size=batch_size, verbose=0)

        for p, (y0, x0) in zip(pred, locations):
            accum[y0:y0+patch_size, x0:x0+patch_size] += p
            count[y0:y0+patch_size, x0:x0+patch_size] += 1.0

        patches.clear()
        locations.clear()

    for y0 in _origins(h, patch_size, stride):
        for x0 in _origins(w, patch_size, stride):
            patches.append(x[y0:y0+patch_size, x0:x0+patch_size])
            locations.append((y0, x0))
            if len(patches) >= batch_size:
                flush()

    flush()

    return (accum / np.maximum(count, 1.0)).astype(np.float32)
