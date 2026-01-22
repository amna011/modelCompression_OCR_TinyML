from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from .config import load_config
from .ctc import Charset, batch_cer, exact_match_accuracy, greedy_decode
from .data import load_and_prepare, load_kaggle_df, split_by_group
from .models import infer_time_steps


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


def _select_eval_split(splits, split_name: str):
    if split_name == "test":
        return splits.test
    if split_name == "val":
        return splits.val
    return splits.train


def _infer_tflite_time_steps(output_details, cfg: dict) -> int:
    shape = output_details["shape"]
    if len(shape) == 3:
        steps = int(shape[1])
    elif len(shape) == 2:
        steps = int(shape[0])
    else:
        raise ValueError("TFLite output shape is not compatible with CTC decoding.")
    if steps < 0:
        max_steps = cfg.get("ctc", {}).get("max_time_steps")
        if max_steps is None:
            raise ValueError("CTC time steps is dynamic; set ctc.max_time_steps explicitly.")
        steps = int(max_steps)
    return steps


def _quantize_input(x: np.ndarray, scale: float, zero_point: int, dtype) -> np.ndarray:
    if scale == 0:
        return x.astype(dtype)
    q = np.round(x / scale + zero_point)
    info = np.iinfo(dtype)
    q = np.clip(q, info.min, info.max)
    return q.astype(dtype)


def _dequantize_output(y: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale == 0:
        return y.astype(np.float32)
    return (y.astype(np.float32) - zero_point) * scale


def _tflite_predict(interpreter: tf.lite.Interpreter, images: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details["index"]
    output_index = output_details["index"]
    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]
    input_scale, input_zero = input_details["quantization"]
    output_scale, output_zero = output_details["quantization"]

    outputs = []
    for idx in range(len(images)):
        sample = images[idx : idx + 1]
        if input_dtype in (np.int8, np.uint8):
            sample = _quantize_input(sample, float(input_scale), int(input_zero), input_dtype)
        else:
            sample = sample.astype(input_dtype)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        if output_dtype in (np.int8, np.uint8):
            output = _dequantize_output(output, float(output_scale), int(output_zero))
        if output.shape[0] == 1:
            output = output[0]
        outputs.append(output)
    return np.stack(outputs, axis=0)


def evaluate(config_path: str, model_path: str, split_name: str) -> None:
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

    eval_samples = _select_eval_split(splits, split_name)

    preds = model.predict(eval_samples.images, batch_size=int(cfg["training"]["batch_size"]), verbose=0)
    decoded = greedy_decode(preds, charset)

    cer = batch_cer(eval_samples.texts, decoded)
    exact = exact_match_accuracy(eval_samples.texts, decoded)

    print(f"CER: {cer:.4f}")
    print(f"Exact Match: {exact:.4f}")


def evaluate_tflite(config_path: str, tflite_path: str, split_name: str) -> None:
    cfg = load_config(config_path)
    charset = Charset(cfg["charset"])

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    input_shape = input_details["shape"]
    if input_shape[0] == -1:
        resized = [1, int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]
        interpreter.resize_tensor_input(input_details["index"], resized, strict=False)
        interpreter.allocate_tensors()

    output_details = interpreter.get_output_details()[0]
    inferred_steps = _infer_tflite_time_steps(output_details, cfg)
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

    eval_samples = _select_eval_split(splits, split_name)
    preds = _tflite_predict(interpreter, eval_samples.images)
    decoded = greedy_decode(preds, charset)

    cer = batch_cer(eval_samples.texts, decoded)
    exact = exact_match_accuracy(eval_samples.texts, decoded)

    print(f"CER: {cer:.4f}")
    print(f"Exact Match: {exact:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR models with CER and exact match.")
    parser.add_argument("--config", required=True, help="Path to config yaml.")
    parser.add_argument("--model_path", help="Path to saved keras model.")
    parser.add_argument("--tflite_path", help="Path to int8 tflite model.")
    parser.add_argument("--split", default="test", choices=["train", "test", "val"])
    args = parser.parse_args()

    if not args.model_path and not args.tflite_path:
        raise SystemExit("Provide --model_path or --tflite_path.")
    if args.tflite_path:
        evaluate_tflite(args.config, args.tflite_path, args.split)
    else:
        evaluate(args.config, args.model_path, args.split)


if __name__ == "__main__":
    main()
