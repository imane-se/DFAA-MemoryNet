"""DFAA-MemoryNet v9.2 architecture and dual-task loss."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(inputs, filters: int, dropout_rate: float):
    """Two 3×3 convolutional layers with batch normalization and swish."""
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-6),
    )(inputs)
    x = layers.BatchNormalization(momentum=0.90)(x)
    x = layers.Activation("swish")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-6),
    )(x)
    x = layers.BatchNormalization(momentum=0.90)(x)
    x = layers.Activation("swish")(x)

    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)

    return x


def build_memorynet(
    input_shape,
    base_filters: int = 24,
    dropout_scale: float = 1.05,
    fraction_output_bias: float = 0.0,
    core_output_bias: float = 0.0,
    name: str = "DFAA_MemoryNet_v9_2",
):
    """Build the reference dual-task DFAA-MemoryNet architecture."""
    inputs = keras.Input(shape=input_shape, name="eo_memory_tensor")

    c1 = conv_block(inputs, base_filters, 0.04 * dropout_scale)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, base_filters * 2, 0.07 * dropout_scale)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, base_filters * 4, 0.10 * dropout_scale)
    p3 = layers.MaxPooling2D()(c3)

    bottleneck = conv_block(
        p3, base_filters * 8, 0.14 * dropout_scale
    )

    u3 = layers.Conv2DTranspose(
        base_filters * 4, 2, strides=2, padding="same"
    )(bottleneck)
    c4 = conv_block(
        layers.Concatenate()([u3, c3]),
        base_filters * 4,
        0.10 * dropout_scale,
    )

    u2 = layers.Conv2DTranspose(
        base_filters * 2, 2, strides=2, padding="same"
    )(c4)
    c5 = conv_block(
        layers.Concatenate()([u2, c2]),
        base_filters * 2,
        0.07 * dropout_scale,
    )

    u1 = layers.Conv2DTranspose(
        base_filters, 2, strides=2, padding="same"
    )(c5)
    shared = conv_block(
        layers.Concatenate()([u1, c1]),
        base_filters,
        0.04 * dropout_scale,
    )

    fraction_head = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        dtype="float32",
        bias_initializer=keras.initializers.Constant(fraction_output_bias),
        name="fraction_head",
    )(shared)

    core_head = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        dtype="float32",
        bias_initializer=keras.initializers.Constant(core_output_bias),
        name="core_head",
    )(shared)

    outputs = layers.Concatenate(name="dualtask_output")(
        [fraction_head, core_head]
    )

    return keras.Model(inputs, outputs, name=name)


def make_dualtask_loss(
    fraction_weight: float = 0.75,
    core_weight: float = 0.25,
    fraction_intensity_weight: float = 4.0,
    event_bce_weight: float = 0.70,
    event_tversky_weight: float = 0.30,
    tversky_alpha: float = 0.45,
    tversky_beta: float = 0.55,
    tversky_gamma: float = 1.0,
    positive_class_weight: float = 1.0,
):
    """Return the intensity-aware dual-task loss used by DFAA-MemoryNet v9.2."""

    def loss(y_true, y_pred):
        target_fraction = tf.cast(y_true[..., 0:1], tf.float32)
        target_core = tf.cast(y_true[..., 1:2], tf.float32)
        mask = tf.cast(y_true[..., 2:3], tf.float32)

        pred_fraction = tf.clip_by_value(
            tf.cast(y_pred[..., 0:1], tf.float32), 1e-6, 1.0 - 1e-6
        )
        pred_core = tf.clip_by_value(
            tf.cast(y_pred[..., 1:2], tf.float32), 1e-6, 1.0 - 1e-6
        )

        fraction_pixel_weight = (
            1.0 + fraction_intensity_weight * tf.square(target_fraction)
        )
        fraction_loss = (
            tf.reduce_sum(
                tf.abs(target_fraction - pred_fraction)
                * fraction_pixel_weight
                * mask
            )
            / (
                tf.reduce_sum(fraction_pixel_weight * mask)
                + 1e-6
            )
        )

        event_pixel_weight = tf.where(
            target_core > 0.5,
            positive_class_weight,
            1.0,
        )
        bce = (
            -target_core * tf.math.log(pred_core)
            - (1.0 - target_core) * tf.math.log(1.0 - pred_core)
        )
        weighted_bce = (
            tf.reduce_sum(bce * event_pixel_weight * mask)
            / (
                tf.reduce_sum(event_pixel_weight * mask)
                + 1e-6
            )
        )

        tp = tf.reduce_sum(target_core * pred_core * mask)
        fp = tf.reduce_sum((1.0 - target_core) * pred_core * mask)
        fn = tf.reduce_sum(target_core * (1.0 - pred_core) * mask)

        tversky = (tp + 1e-6) / (
            tp
            + tversky_alpha * fp
            + tversky_beta * fn
            + 1e-6
        )
        tversky_loss = tf.pow(1.0 - tversky, tversky_gamma)

        event_loss = (
            event_bce_weight * weighted_bce
            + event_tversky_weight * tversky_loss
        )

        return (
            fraction_weight * fraction_loss
            + core_weight * event_loss
        )

    loss.__name__ = "memorysafe_precisionaware_v9_2_dualtask_loss"
    return loss


def masked_fraction_mae(y_true, y_pred):
    target = tf.cast(y_true[..., 0:1], tf.float32)
    mask = tf.cast(y_true[..., 2:3], tf.float32)
    prediction = tf.cast(y_pred[..., 0:1], tf.float32)
    return (
        tf.reduce_sum(tf.abs(target - prediction) * mask)
        / (tf.reduce_sum(mask) + 1e-6)
    )


@keras.utils.register_keras_serializable(package="DFAA")
class MaskedCorePRAUC(keras.metrics.Metric):
    def __init__(self, name="core_pr_auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = keras.metrics.AUC(curve="PR")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.auc.update_state(
            tf.cast(y_true[..., 1], tf.float32),
            tf.cast(y_pred[..., 1], tf.float32),
            sample_weight=tf.cast(y_true[..., 2], tf.float32),
        )

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_state()


@keras.utils.register_keras_serializable(package="DFAA")
class MaskedCoreROCAUC(keras.metrics.Metric):
    def __init__(self, name="core_roc_auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = keras.metrics.AUC(curve="ROC")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.auc.update_state(
            tf.cast(y_true[..., 1], tf.float32),
            tf.cast(y_pred[..., 1], tf.float32),
            sample_weight=tf.cast(y_true[..., 2], tf.float32),
        )

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_state()
