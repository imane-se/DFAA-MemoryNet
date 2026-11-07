"""Training helpers for DFAA-MemoryNet.

This module preserves the reference v9.2 optimizer, output-bias initialization,
positive-class weighting, and training-only normalization. The manuscript's
study-specific anti-collapse model-selection callback is intentionally outside
this model-only repository.
"""

from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .memorynet import (
    build_memorynet,
    make_dualtask_loss,
    masked_fraction_mae,
    MaskedCorePRAUC,
    MaskedCoreROCAUC,
)
from .predictors import fit_normalizer, apply_normalizer


def training_initialization(y_train: np.ndarray) -> dict:
    """Derive the reference v9.2 output biases and class weight from training data.

    The training target tensor is expected to contain:
    channel 0 = four-week hotspot fraction,
    channel 1 = binary persistent-event target,
    channel 2 = valid-pixel mask.
    """
    y_train = np.asarray(y_train)
    if y_train.ndim < 2 or y_train.shape[-1] < 3:
        raise ValueError("y_train must have at least 3 target channels.")

    valid = y_train[..., 2] > 0
    if not np.any(valid):
        raise ValueError("No valid training target pixels.")

    event_prevalence = float(np.mean(y_train[..., 1][valid] > 0))
    fraction_mean = float(np.mean(y_train[..., 0][valid]))

    event_prevalence = float(np.clip(event_prevalence, 1e-4, 1.0 - 1e-4))
    fraction_mean = float(np.clip(fraction_mean, 1e-4, 1.0 - 1e-4))

    positive_class_weight = float(
        np.clip(
            np.sqrt((1.0 - event_prevalence) / event_prevalence),
            1.0,
            3.0,
        )
    )

    fraction_output_bias = float(
        np.log(fraction_mean / (1.0 - fraction_mean))
    )
    core_output_bias = float(
        np.log(event_prevalence / (1.0 - event_prevalence))
    )

    return {
        "event_prevalence": event_prevalence,
        "fraction_mean": fraction_mean,
        "positive_class_weight": positive_class_weight,
        "fraction_output_bias": fraction_output_bias,
        "core_output_bias": core_output_bias,
    }


def compile_model(
    model,
    learning_rate: float = 2e-4,
    positive_class_weight: float = 1.0,
    weight_decay: float = 2e-5,
    clipnorm: float = 1.0,
):
    """Compile DFAA-MemoryNet with the reference v9.2 optimizer and loss."""
    loss = make_dualtask_loss(
        positive_class_weight=positive_class_weight,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        ),
        loss=loss,
        metrics=[
            masked_fraction_mae,
            MaskedCorePRAUC(),
            MaskedCoreROCAUC(),
        ],
    )
    return model


def train_memorynet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    base_filters: int = 24,
    dropout_scale: float = 1.05,
    learning_rate: float = 2e-4,
    epochs: int = 60,
    batch_size: int = 4,
    seed: int = 42,
):
    """Fit the reference DFAA-MemoryNet model.

    Normalization statistics are estimated strictly from training data.
    Validation data, when supplied, are normalized with those same statistics.

    Notes
    -----
    The manuscript workflow additionally used a study-specific anti-collapse
    selection callback and held-out-year logic. Those experiment-control
    routines are not part of this model-only package.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    mean, std = fit_normalizer(x_train)
    x_train_n = apply_normalizer(x_train, mean, std)

    init = training_initialization(y_train)

    validation_data = None
    if x_val is not None and y_val is not None:
        x_val_n = apply_normalizer(x_val, mean, std)
        validation_data = (x_val_n, y_val)

    model = build_memorynet(
        input_shape=x_train.shape[1:],
        base_filters=base_filters,
        dropout_scale=dropout_scale,
        fraction_output_bias=init["fraction_output_bias"],
        core_output_bias=init["core_output_bias"],
    )

    compile_model(
        model,
        learning_rate=learning_rate,
        positive_class_weight=init["positive_class_weight"],
    )

    callbacks = [keras.callbacks.TerminateOnNaN()]

    if validation_data is not None:
        callbacks.extend([
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_delta=1e-4,
                min_lr=1e-6,
                verbose=1,
            ),
        ])

    history = model.fit(
        x_train_n,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history, mean, std, init


def save_normalization(
    path: str | Path,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: list[str],
):
    """Save normalization statistics and exact feature order."""
    np.savez_compressed(
        path,
        mean=mean,
        std=std,
        feature_names=np.asarray(feature_names, dtype=object),
    )
