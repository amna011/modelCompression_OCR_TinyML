from __future__ import annotations

import argparse
import os

import tensorflow as tf

from .config import load_config
from .ctc import Charset
from .data import load_and_prepare, load_kaggle_df, make_tf_dataset, split_by_group
from .models import CTCModel, build_model, infer_time_steps


def train(config_path: str, model_name: str, output_path: str | None) -> str:
    cfg = load_config(config_path)
    charset = Charset(cfg["charset"])
    input_shape = (cfg["image"]["height"], cfg["image"]["width"], 1)
    tf.random.set_seed(int(cfg["seed"]))

    base_model = build_model(model_name, input_shape, charset, cfg)
    inferred_steps = infer_time_steps(base_model)
    ctc_max_steps = cfg.get("ctc", {}).get("max_time_steps")
    if ctc_max_steps is None:
        ctc_max_steps = inferred_steps
    else:
        ctc_max_steps = int(ctc_max_steps)
        if ctc_max_steps > inferred_steps:
            print(
                f"Warning: ctc.max_time_steps ({ctc_max_steps}) > model time steps "
                f"({inferred_steps}); using {inferred_steps}."
            )
            ctc_max_steps = inferred_steps

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

    model = CTCModel(base_model)
    lr_key = f"learning_rate_{model_name}"
    learning_rate = cfg["training"].get(lr_key, cfg["training"]["learning_rate"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(cfg["training"]["early_stop_patience"]),
            restore_best_weights=True,
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["training"]["epochs"]),
        callbacks=callbacks,
    )

    os.makedirs("models", exist_ok=True)
    if not output_path:
        output_path = os.path.join("models", f"{model_name}_fp32.keras")
    base_model.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OCR models with CTC loss.")
    parser.add_argument("--config", required=True, help="Path to config yaml.")
    parser.add_argument("--model", required=True, choices=["lightweight_cnn", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--out", default="", help="Output model path.")
    args = parser.parse_args()

    output = train(args.config, args.model, args.out or None)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()
