from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import tensorflow as tf

from .ctc import Charset


@dataclass
class ModelSpec:
    name: str
    input_shape: Tuple[int, int, int]
    num_classes: int


def _relu_layer(use_relu6: bool) -> tf.keras.layers.Layer:
    if use_relu6:
        return tf.keras.layers.ReLU(max_value=6.0)
    return tf.keras.layers.ReLU()


def _select_backbone_output(
    backbone: tf.keras.Model, feature_layer: str | None
) -> tf.keras.Model:
    if not feature_layer:
        return backbone
    try:
        output = backbone.get_layer(feature_layer).output
    except ValueError as exc:
        raise ValueError(f"Feature layer '{feature_layer}' not found in {backbone.name}.") from exc
    return tf.keras.Model(inputs=backbone.input, outputs=output, name=f"{backbone.name}_feat")


def build_lightweight_cnn_with_filters(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    filters: Tuple[int, ...],
    use_relu6: bool,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = inputs

    for idx, filters_count in enumerate(filters):
        strides = (2, 2) if idx < 3 else (2, 1)
        x = tf.keras.layers.SeparableConv2D(
            filters_count,
            kernel_size=3,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"sep_conv_{idx}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{idx}")(x)
        x = _relu_layer(use_relu6)(x)

    # Reduce height to 1 while keeping width as time steps.
    height = x.shape[1]
    if height is None:
        x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(height, 1))(x)
    # Use 1x1 conv to avoid Dense->Tensordot (adds REDUCE_PROD in TFLite).
    x = tf.keras.layers.Conv2D(num_classes, kernel_size=1, padding="same", name="logits")(x)
    x = tf.squeeze(x, axis=1)
    x = tf.keras.layers.Softmax(name="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="lightweight_cnn")


def build_lightweight_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    base_filters: int,
    use_relu6: bool,
) -> tf.keras.Model:
    filters = (
        base_filters,
        base_filters * 2,
        base_filters * 4,
        base_filters * 6,
        base_filters * 6,
        base_filters * 6,
    )
    return build_lightweight_cnn_with_filters(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=filters,
        use_relu6=use_relu6,
    )


def build_mobilenet_v2(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    alpha: float,
    feature_layer: str | None,
    pretrained: bool,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = inputs
    if input_shape[-1] == 1:
        x = tf.keras.layers.Conv2D(
            3, kernel_size=1, padding="same", use_bias=False, name="rgb_stem"
        )(x)

    weights = "imagenet" if pretrained else None
    try:
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(input_shape[0], input_shape[1], 3),
            alpha=alpha,
            include_top=False,
            weights=weights,
        )
    except Exception as exc:
        if not pretrained:
            raise
        print(f"Warning: failed to load MobileNetV2 pretrained weights: {exc}")
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(input_shape[0], input_shape[1], 3),
            alpha=alpha,
            include_top=False,
            weights=None,
        )
    backbone = _select_backbone_output(backbone, feature_layer)
    x = backbone(x)
    height = x.shape[1]
    if height is None:
        x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(height, 1))(x)
    x = tf.keras.layers.Conv2D(num_classes, kernel_size=1, padding="same", name="logits")(x)
    x = tf.squeeze(x, axis=1)
    x = tf.keras.layers.Softmax(name="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="mobilenet_v2")


def build_efficientnet_b0(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    feature_layer: str | None,
    pretrained: bool,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = inputs
    if input_shape[-1] == 1:
        x = tf.keras.layers.Conv2D(
            3, kernel_size=1, padding="same", use_bias=False, name="rgb_stem"
        )(x)

    weights = "imagenet" if pretrained else None
    try:
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=(input_shape[0], input_shape[1], 3),
            include_top=False,
            weights=weights,
        )
    except Exception as exc:
        if not pretrained:
            raise
        print(f"Warning: failed to load EfficientNetB0 pretrained weights: {exc}")
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=(input_shape[0], input_shape[1], 3),
            include_top=False,
            weights=None,
        )
    backbone = _select_backbone_output(backbone, feature_layer)
    x = backbone(x)
    height = x.shape[1]
    if height is None:
        x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(height, 1))(x)
    x = tf.keras.layers.Conv2D(num_classes, kernel_size=1, padding="same", name="logits")(x)
    x = tf.squeeze(x, axis=1)
    x = tf.keras.layers.Softmax(name="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="efficientnet_b0")


def infer_time_steps(model: tf.keras.Model) -> int:
    shape = model.output_shape
    if isinstance(shape, list):
        shape = shape[0]
    if not shape or len(shape) < 3:
        raise ValueError("Model output shape is not compatible with CTC decoding.")
    steps = shape[1]
    if steps is None:
        raise ValueError("CTC time steps is dynamic; set ctc.max_time_steps explicitly.")
    return int(steps)


class CTCModel(tf.keras.Model):
    def __init__(self, base_model: tf.keras.Model):
        super().__init__()
        self.base_model = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        inputs, labels = data
        images = inputs["image"]
        label_length = tf.cast(inputs["label_length"], tf.int32)

        with tf.GradientTape() as tape:
            y_pred = self(images, training=True)
            input_length = tf.fill([tf.shape(y_pred)[0], 1], tf.shape(y_pred)[1])
            loss = tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        inputs, labels = data
        images = inputs["image"]
        label_length = tf.cast(inputs["label_length"], tf.int32)
        y_pred = self(images, training=False)
        input_length = tf.fill([tf.shape(y_pred)[0], 1], tf.shape(y_pred)[1])
        loss = tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
        loss = tf.reduce_mean(loss)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def build_model(
    name: str,
    input_shape: Tuple[int, int, int],
    charset: Charset,
    cfg: Dict,
) -> tf.keras.Model:
    num_classes = charset.num_classes
    if name == "lightweight_cnn":
        return build_lightweight_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            base_filters=int(cfg["lightweight_cnn"]["base_filters"]),
            use_relu6=bool(cfg["lightweight_cnn"].get("use_relu6", True)),
        )
    if name == "mobilenet_v2":
        return build_mobilenet_v2(
            input_shape=input_shape,
            num_classes=num_classes,
            alpha=float(cfg["mobilenet_v2"]["alpha"]),
            feature_layer=cfg["mobilenet_v2"].get("feature_layer"),
            pretrained=bool(cfg["mobilenet_v2"].get("pretrained", False)),
        )
    if name == "efficientnet_b0":
        return build_efficientnet_b0(
            input_shape=input_shape,
            num_classes=num_classes,
            feature_layer=cfg["efficientnet"].get("feature_layer"),
            pretrained=bool(cfg["efficientnet"].get("pretrained", False)),
        )
    raise ValueError(f"Unknown model name: {name}")
