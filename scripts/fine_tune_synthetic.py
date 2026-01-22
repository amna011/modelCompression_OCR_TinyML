#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import tensorflow as tf

from ocr.config import load_config
from ocr.ctc import Charset
from ocr.data import load_and_prepare, load_kaggle_df, make_tf_dataset, split_by_group
from ocr.models import CTCModel, infer_time_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune OCR model on synthetic plates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--dataset_file", default="plates.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    os.environ["KAGGLE_DATASET_ROOT"] = args.dataset_root
    os.environ["KAGGLE_DATASET_FILE"] = args.dataset_file

    cfg = load_config(args.config)
    charset = Charset(cfg["charset"])

    base_model = tf.keras.models.load_model(args.base_model)
    inferred_steps = infer_time_steps(base_model)
    ctc_max_steps = cfg.get("ctc", {}).get("max_time_steps")
    if ctc_max_steps is None:
        ctc_max_steps = inferred_steps
    else:
        ctc_max_steps = int(ctc_max_steps)
        if ctc_max_steps > inferred_steps:
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
        batch_size=args.batch_size,
        augment=True,
        augment_cfg=cfg["augmentation"],
        shuffle=True,
    )
    val_ds = make_tf_dataset(
        splits.val,
        batch_size=args.batch_size,
        augment=False,
        augment_cfg=cfg["augmentation"],
        shuffle=False,
    )

    model = CTCModel(base_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
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
        epochs=args.epochs,
        callbacks=callbacks,
    )

    base_model.save(args.out)
    print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
