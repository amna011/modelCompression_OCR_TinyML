from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import tensorflow as tf

from .config import load_config
from .ctc import Charset
from .data import load_and_prepare, load_kaggle_df, make_tf_dataset, split_by_group
from .models import CTCModel, build_model, infer_time_steps


@dataclass
class FeaturePair:
    teacher_layer: str
    student_layer: str


def _resolve_feature_pairs(teacher: tf.keras.Model, student: tf.keras.Model, names: List[str]) -> List[FeaturePair]:
    pairs: List[FeaturePair] = []
    for name in names:
        if name in [l.name for l in teacher.layers] and name in [l.name for l in student.layers]:
            pairs.append(FeaturePair(name, name))
    return pairs


class DistillModel(CTCModel):
    def __init__(
        self,
        student_base: tf.keras.Model,
        teacher_base: tf.keras.Model,
        feature_pairs: List[FeaturePair],
        temperature: float,
        alpha: float,
        feature_weight: float = 0.1,
    ):
        super().__init__(student_base)
        self.teacher = teacher_base
        self.teacher.trainable = False
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.feature_pairs = feature_pairs

        if feature_pairs:
            teacher_outputs = [teacher_base.get_layer(p.teacher_layer).output for p in feature_pairs]
            self.teacher_features = tf.keras.Model(
                inputs=teacher_base.inputs,
                outputs=teacher_outputs + [teacher_base.output],
            )
            student_outputs = [student_base.get_layer(p.student_layer).output for p in feature_pairs]
            self.student_features = tf.keras.Model(
                inputs=student_base.inputs,
                outputs=student_outputs + [student_base.output],
            )
        else:
            self.teacher_features = None
            self.student_features = None

    def _feature_loss(self, student_feats: List[tf.Tensor], teacher_feats: List[tf.Tensor]) -> tf.Tensor:
        loss = 0.0
        for s, t in zip(student_feats, teacher_feats):
            if len(s.shape) > 2:
                s = tf.reduce_mean(s, axis=list(range(1, len(s.shape) - 1)))
            if len(t.shape) > 2:
                t = tf.reduce_mean(t, axis=list(range(1, len(t.shape) - 1)))
            if s.shape[-1] != t.shape[-1]:
                continue
            loss += tf.reduce_mean(tf.square(s - t))
        return loss

    def train_step(self, data):
        inputs, labels = data
        images = inputs["image"]
        label_length = tf.cast(inputs["label_length"], tf.int32)

        with tf.GradientTape() as tape:
            if self.teacher_features and self.student_features:
                teacher_outputs = self.teacher_features(images, training=False)
                student_outputs = self.student_features(images, training=True)
                teacher_probs = teacher_outputs[-1]
                student_probs = student_outputs[-1]
                teacher_feats = teacher_outputs[:-1]
                student_feats = student_outputs[:-1]
            else:
                teacher_probs = self.teacher(images, training=False)
                student_probs = self.base_model(images, training=True)
                teacher_feats = []
                student_feats = []

            input_length = tf.fill([tf.shape(student_probs)[0], 1], tf.shape(student_probs)[1])
            ctc = tf.keras.backend.ctc_batch_cost(labels, student_probs, input_length, label_length)
            ctc = tf.reduce_mean(ctc)

            t = self.temperature
            teacher_logits = tf.math.log(tf.clip_by_value(teacher_probs, 1e-7, 1.0))
            student_logits = tf.math.log(tf.clip_by_value(student_probs, 1e-7, 1.0))
            teacher_soft = tf.nn.softmax(teacher_logits / t)
            student_soft = tf.nn.softmax(student_logits / t)
            distill = tf.reduce_mean(tf.keras.losses.kl_divergence(teacher_soft, student_soft))

            feat_loss = self._feature_loss(student_feats, teacher_feats) if teacher_feats else 0.0
            loss = self.alpha * ctc + (1.0 - self.alpha) * distill + self.feature_weight * feat_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def distill(
    config_path: str,
    teacher_path: str,
    student_name: str,
    output_path: str,
) -> str:
    cfg = load_config(config_path)
    charset = Charset(cfg["charset"])
    input_shape = (cfg["image"]["height"], cfg["image"]["width"], 1)
    student_base = build_model(student_name, input_shape, charset, cfg)
    inferred_steps = infer_time_steps(student_base)
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

    teacher_base = tf.keras.models.load_model(teacher_path)

    feature_names = cfg.get("distillation", {}).get("feature_layer_names", [])
    pairs = _resolve_feature_pairs(teacher_base, student_base, feature_names)

    model = DistillModel(
        student_base=student_base,
        teacher_base=teacher_base,
        feature_pairs=pairs,
        temperature=float(cfg["distillation"]["temperature"]),
        alpha=float(cfg["distillation"]["alpha"]),
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["training"]["learning_rate"])))
    model.fit(train_ds, validation_data=val_ds, epochs=int(cfg["training"]["epochs"]))

    student_base.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge distillation training for student OCR models.")
    parser.add_argument("--config", required=True, help="Path to config yaml.")
    parser.add_argument("--teacher", required=True, help="Path to teacher keras model.")
    parser.add_argument("--student", required=True, choices=["lightweight_cnn", "mobilenet_v2"])
    parser.add_argument("--out", required=True, help="Output student model path.")
    args = parser.parse_args()

    path = distill(args.config, args.teacher, args.student, args.out)
    print(f"Saved distilled model to {path}")


if __name__ == "__main__":
    main()
