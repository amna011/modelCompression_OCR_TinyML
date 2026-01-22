from __future__ import annotations

import argparse
import os
from typing import Iterator, List

import numpy as np
import tensorflow as tf

from .config import load_config
from .ctc import Charset
from .data import load_and_prepare, load_kaggle_df, make_tf_dataset, split_by_group
from .models import CTCModel, infer_time_steps


def _representative_dataset(images: np.ndarray, max_samples: int = 200):
    def _gen() -> Iterator[List[np.ndarray]]:
        count = min(len(images), max_samples)
        for i in range(count):
            yield [images[i : i + 1]]

    return _gen


def _convert_to_int8(model: tf.keras.Model, rep_data: Iterator[List[np.ndarray]]) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def _check_int8_only(tflite_path: str) -> None:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    details = interpreter.get_input_details() + interpreter.get_output_details()
    for detail in details:
        if detail["dtype"] != np.int8:
            raise ValueError("Model is not fully int8.")


def _resolve_ctc_max_steps(cfg: dict, inferred_steps: int) -> int:
    ctc_max_steps = cfg.get("ctc", {}).get("max_time_steps")
    if ctc_max_steps is None:
        return inferred_steps
    ctc_max_steps = int(ctc_max_steps)
    if ctc_max_steps > inferred_steps:
        print(
            f"Warning: ctc.max_time_steps ({ctc_max_steps}) > model time steps "
            f"({inferred_steps}); using {inferred_steps}."
        )
        return inferred_steps
    return ctc_max_steps


def ptq_convert(config_path: str, model_path: str, out_path: str) -> str:
    cfg = load_config(config_path)
    charset = Charset(cfg["charset"])

    model = tf.keras.models.load_model(model_path)
    inferred_steps = infer_time_steps(model)
    ctc_max_steps = _resolve_ctc_max_steps(cfg, inferred_steps)

    df = load_kaggle_df()
    samples = load_and_prepare(
        df,
        charset=charset,
        label_max_length=int(cfg["training"]["label_max_length"]),
        target_h=int(cfg["image"]["height"]),
        target_w=int(cfg["image"]["width"]),
        keep_aspect=bool(cfg["image"]["keep_aspect"]),
        group_column=cfg["split"].get("group_column") or None,
        group_from_path_regex=cfg["split"].get("group_from_path_regex") or None,
        ctc_max_time_steps=ctc_max_steps,
    )
    splits = split_by_group(
        samples,
        train_ratio=float(cfg["split"]["train"]),
        test_ratio=float(cfg["split"]["test"]),
        val_ratio=float(cfg["split"]["val"]),
        seed=int(cfg["seed"]),
    )

    rep_data = _representative_dataset(splits.val.images)
    tflite = _convert_to_int8(model, rep_data)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite)

    _check_int8_only(out_path)
    return out_path


def qat_convert(
    config_path: str,
    model_path: str,
    out_path: str,
    fine_tune_epochs: int,
) -> str:
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError as exc:
        raise ImportError(
            "QAT requires tensorflow_model_optimization with a compatible Keras install. "
            "Install `tf_keras` or disable TF_USE_LEGACY_KERAS."
        ) from exc

    cfg = load_config(config_path)
    charset = Charset(cfg["charset"])

    base_model = tf.keras.models.load_model(model_path)
    inferred_steps = infer_time_steps(base_model)
    ctc_max_steps = _resolve_ctc_max_steps(cfg, inferred_steps)

    df = load_kaggle_df()
    samples = load_and_prepare(
        df,
        charset=charset,
        label_max_length=int(cfg["training"]["label_max_length"]),
        target_h=int(cfg["image"]["height"]),
        target_w=int(cfg["image"]["width"]),
        keep_aspect=bool(cfg["image"]["keep_aspect"]),
        group_column=cfg["split"].get("group_column") or None,
        group_from_path_regex=cfg["split"].get("group_from_path_regex") or None,
        ctc_max_time_steps=ctc_max_steps,
    )
    splits = split_by_group(
        samples,
        train_ratio=float(cfg["split"]["train"]),
        test_ratio=float(cfg["split"]["test"]),
        val_ratio=float(cfg["split"]["val"]),
        seed=int(cfg["seed"]),
    )

    train_ds = make_tf_dataset(
        splits.train,
        batch_size=int(cfg["training"]["batch_size"]),
        augment=bool(cfg["augmentation"]["enable"]),
        augment_cfg=cfg["augmentation"],
        shuffle=True,
    )
    val_ds = make_tf_dataset(
        splits.val,
        batch_size=int(cfg["training"]["batch_size"]),
        augment=False,
        augment_cfg=cfg["augmentation"],
        shuffle=False,
    )

    quantized_base = tfmot.quantization.keras.quantize_model(base_model)
    quant_model = CTCModel(quantized_base)
    quant_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["training"]["learning_rate"])),
    )

    quant_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
    )

    rep_data = _representative_dataset(splits.val.images)
    tflite = _convert_to_int8(quant_model.base_model, rep_data)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite)

    _check_int8_only(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="PTQ or QAT convert to int8 TFLite.")
    parser.add_argument("--config", required=True, help="Path to config yaml.")
    parser.add_argument("--model_path", required=True, help="Path to saved keras model.")
    parser.add_argument("--out", required=True, help="Output tflite path.")
    parser.add_argument("--qat", action="store_true", help="Run QAT before conversion.")
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    args = parser.parse_args()

    if args.qat:
        path = qat_convert(args.config, args.model_path, args.out, args.fine_tune_epochs)
    else:
        path = ptq_convert(args.config, args.model_path, args.out)
    print(f"Saved int8 model to {path}")


if __name__ == "__main__":
    main()
