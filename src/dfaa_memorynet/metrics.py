"""Evaluation metrics used with DFAA-MemoryNet."""

from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def exact_top_fraction_mask(
    score: np.ndarray,
    valid: np.ndarray,
    fraction: float = 0.15,
) -> np.ndarray:
    """Select exactly the top spatial fraction of valid pixels."""
    valid = np.asarray(valid, dtype=bool) & np.isfinite(score)
    ids = np.flatnonzero(valid.ravel())
    out = np.zeros(score.size, dtype=bool)

    if ids.size == 0:
        return out.reshape(score.shape)

    k = max(1, int(np.floor(float(fraction) * ids.size)))
    values = np.asarray(score).ravel()[ids]
    order = np.argsort(values, kind="mergesort")
    out[ids[order[-k:]]] = True
    return out.reshape(score.shape)


def spatial_binary_metrics(
    observed: np.ndarray,
    selected: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """Precision, recovery/recall, CSI, F1 and F2."""
    m = np.asarray(valid, dtype=bool)
    obs = np.asarray(observed, dtype=bool)[m]
    pred = np.asarray(selected, dtype=bool)[m]

    tp = int(np.sum(obs & pred))
    fp = int(np.sum((~obs) & pred))
    fn = int(np.sum(obs & (~pred)))

    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision + recall) and precision + recall > 0
        else np.nan
    )
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if np.isfinite(precision + recall) and (4 * precision + recall) > 0
        else np.nan
    )

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision,
        "recall": recall,
        "recovery": recall,
        "CSI": csi,
        "F1": f1,
        "F2": f2,
    }


def evaluate_continuous_score(
    observed_binary: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
    top_fraction: float = 0.15,
) -> dict:
    """PR-AUC/ROC-AUC/Brier plus exact equal-area spatial metrics."""
    common = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(score)
        & np.isfinite(observed_binary)
    )
    y = np.asarray(observed_binary)[common].astype(np.uint8)
    s = np.asarray(score)[common].astype(np.float64)

    if y.size == 0 or np.unique(y).size < 2:
        raise ValueError("Evaluation requires both positive and negative observations.")

    selected = exact_top_fraction_mask(score, common, top_fraction)
    spatial = spatial_binary_metrics(observed_binary, selected, common)

    return {
        "PR-AUC": float(average_precision_score(y, s)),
        "ROC-AUC": float(roc_auc_score(y, s)),
        "Brier": float(brier_score_loss(y, np.clip(s, 0, 1))),
        **spatial,
        "n_valid": int(common.sum()),
        "prevalence": float(y.mean()),
    }
