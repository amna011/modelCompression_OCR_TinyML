from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import tensorflow as tf

from .models import build_lightweight_cnn_with_filters


def _sorted_layers(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    sep_layers = [l for l in model.layers if l.name.startswith("sep_conv_")]
    sep_layers.sort(key=lambda l: int(l.name.split("_")[-1]))
    return sep_layers


def _compute_keep_indices(layer: tf.keras.layers.Layer, prune_ratio: float, min_filters: int) -> np.ndarray:
    weights = layer.get_weights()
    if len(weights) < 2:
        raise ValueError("SeparableConv2D weights not found.")
    pointwise = weights[1]
    scores = np.sum(np.abs(pointwise), axis=(0, 1, 2))
    keep_count = max(min_filters, int(round(pointwise.shape[-1] * (1.0 - prune_ratio))))
    keep_idx = np.argsort(scores)[-keep_count:]
    return np.sort(keep_idx)


def prune_lightweight_cnn(
    model_path: str,
    out_path: str,
    prune_ratio: float,
    min_filters: int,
    use_relu6: bool,
) -> str:
    base_model = tf.keras.models.load_model(model_path)
    sep_layers = _sorted_layers(base_model)
    if not sep_layers:
        raise ValueError("No separable conv layers found for pruning.")

    keep_indices = [
        _compute_keep_indices(layer, prune_ratio, min_filters) for layer in sep_layers
    ]
    new_filters = tuple(len(idx) for idx in keep_indices)

    input_shape = base_model.input_shape[1:]
    num_classes = base_model.output_shape[-1]
    new_model = build_lightweight_cnn_with_filters(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=new_filters,
        use_relu6=use_relu6,
    )

    prev_keep = np.arange(input_shape[-1], dtype=np.int64)
    for i, layer in enumerate(sep_layers):
        new_layer = new_model.get_layer(layer.name)
        old_dw, old_pw = layer.get_weights()
        keep = keep_indices[i]

        pruned_dw = old_dw[:, :, prev_keep, :]
        pruned_pw = old_pw[:, :, prev_keep, :]
        pruned_pw = pruned_pw[:, :, :, keep]
        new_layer.set_weights([pruned_dw, pruned_pw])

        bn_old = base_model.get_layer(f"bn_{i}")
        bn_new = new_model.get_layer(f"bn_{i}")
        bn_weights = bn_old.get_weights()
        bn_weights = [w[keep] for w in bn_weights]
        bn_new.set_weights(bn_weights)

        prev_keep = keep

    dense_old = base_model.get_layer("logits")
    dense_new = new_model.get_layer("logits")
    dense_weights = dense_old.get_weights()
    kernel = dense_weights[0][prev_keep, :]
    if len(dense_weights) == 2:
        bias = dense_weights[1]
        dense_new.set_weights([kernel, bias])
    else:
        dense_new.set_weights([kernel])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    new_model.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured pruning for lightweight_cnn.")
    parser.add_argument("--model_path", required=True, help="Path to saved keras model.")
    parser.add_argument("--out", required=True, help="Output pruned model path.")
    parser.add_argument("--prune_ratio", type=float, default=0.3)
    parser.add_argument("--min_filters", type=int, default=8)
    parser.add_argument("--use_relu6", action="store_true", help="Use ReLU6 activations.")
    args = parser.parse_args()

    path = prune_lightweight_cnn(
        args.model_path,
        args.out,
        args.prune_ratio,
        args.min_filters,
        args.use_relu6,
    )
    print(f"Saved pruned model to {path}")


if __name__ == "__main__":
    main()
