"""Training helpers for DFAA-MemoryNet."""

from __future__ import annotations

from pathlib import Path
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


def compile_model(
    model,
    learning_rate: float = 2e-4,
    positive_class_weight: float = 1.0,
):
    """Compile a DFAA-MemoryNet model with the reference v9.2 loss."""
    loss = make_dualtask_loss(
        positive_class_weight=positive_class_weight,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
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
    positive_class_weight: float = 1.0,
    epochs: int = 60,
    batch_size: int = 4,
    seed: int = 42,
):
    """Fit DFAA-MemoryNet and return model, history, mean and std.

    Normalization statistics are estimated from the training data only.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    mean, std = fit_normalizer(x_train)
    x_train_n = apply_normalizer(x_train, mean, std)

    validation_data = None
    if x_val is not None and y_val is not None:
        x_val_n = apply_normalizer(x_val, mean, std)
        validation_data = (x_val_n, y_val)

    model = build_memorynet(
        input_shape=x_train.shape[1:],
        base_filters=base_filters,
        dropout_scale=dropout_scale,
    )
    compile_model(
        model,
        learning_rate=learning_rate,
        positive_class_weight=positive_class_weight,
    )

    callbacks = []
    if validation_data is not None:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

    history = model.fit(
        x_train_n,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history, mean, std


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
