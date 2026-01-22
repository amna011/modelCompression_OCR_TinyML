from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

import kagglehub
from kagglehub.datasets import dataset_download

try:
    from kagglehub import KaggleDatasetAdapter
except ImportError:  # pragma: no cover
    KaggleDatasetAdapter = None

from .ctc import Charset, encode_label


IMAGE_CANDIDATES = [
    "image",
    "images",
    "img",
    "file",
    "path",
    "image_path",
    "img_path",
    "filename",
    "filepath",
]
LABEL_CANDIDATES = [
    "label",
    "labels",
    "text",
    "plate",
    "plate_text",
    "string",
    "plate_number",
]
GROUP_CANDIDATES = ["vehicle", "vehicle_id", "car_id", "track_id", "id"]

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".xml",
    ".parquet",
    ".feather",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".db3",
    ".s3db",
    ".dl3",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".odf",
    ".ods",
    ".odt",
}
PREFERRED_KEYWORDS = ("label", "plate", "text", "annotation", "ocr")


def _read_local_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".json":
        return pd.read_json(path)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    if ext == ".xml":
        return pd.read_xml(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".feather":
        return pd.read_feather(path)
    if ext in {".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file extension for local load: {ext}")


@dataclass
class LoadedSamples:
    images: np.ndarray
    labels: np.ndarray
    label_lengths: np.ndarray
    texts: List[str]
    groups: List[str]


@dataclass
class DatasetSplits:
    train: LoadedSamples
    test: LoadedSamples
    val: LoadedSamples


def _select_dataset_file(dataset_root: str) -> str:
    env_file = os.getenv("KAGGLE_DATASET_FILE", "")
    if env_file:
        if os.path.isabs(env_file):
            return env_file
        return os.path.join(dataset_root, env_file)

    candidates: List[str] = []
    for root, _, files in os.walk(dataset_root):
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue
            candidates.append(os.path.join(root, fname))

    if not candidates:
        raise FileNotFoundError(
            f"No supported dataset files found in {dataset_root}. "
            "Set KAGGLE_DATASET_FILE to a specific file inside the dataset."
        )

    def _score(path: str) -> Tuple[int, int]:
        name = os.path.basename(path).lower()
        keyword_hits = sum(1 for k in PREFERRED_KEYWORDS if k in name)
        return (keyword_hits, -len(name))

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def load_kaggle_df() -> pd.DataFrame:
    if KaggleDatasetAdapter is None:
        raise ImportError(
            "KaggleDatasetAdapter not found in kagglehub. "
            "Upgrade kagglehub (e.g., `pip install -U kagglehub`) and retry."
        )

    dataset_root = os.getenv("KAGGLE_DATASET_ROOT", "").strip()
    if dataset_root:
        dataset_file = _select_dataset_file(dataset_root)
        return _read_local_dataframe(dataset_file)

    try:
        dataset_root = dataset_download("nickyazdani/license-plate-text-recognition-dataset")
    except Exception as exc:
        raise RuntimeError(
            "Failed to download Kaggle dataset. Ensure Kaggle credentials and network access "
            "or set KAGGLE_DATASET_ROOT to a local dataset directory."
        ) from exc

    dataset_file = _select_dataset_file(dataset_root)
    rel_path = os.path.relpath(dataset_file, dataset_root)
    rel_path = rel_path.replace(os.sep, "/")

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "nickyazdani/license-plate-text-recognition-dataset",
        rel_path,
    )
    return df


def _infer_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def _infer_group_from_path(path: str, regex: str) -> Optional[str]:
    if not path or not regex:
        return None
    match = re.search(regex, path)
    if not match:
        return None
    if match.groups():
        return match.group(1)
    return match.group(0)


def _load_image(value, dataset_root: str | None) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return Image.fromarray(value.astype(np.uint8), mode="L")
        if value.ndim == 3:
            return Image.fromarray(value.astype(np.uint8))
    if isinstance(value, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(value))
        except Exception:
            return None
    if isinstance(value, str):
        path = value
        if dataset_root and not os.path.isabs(path):
            path = os.path.join(dataset_root, path)
        if not os.path.exists(path):
            return None
        try:
            return Image.open(path)
        except Exception:
            return None
    return None


def _preprocess_pil(
    img: Image.Image,
    target_h: int,
    target_w: int,
    keep_aspect: bool,
) -> np.ndarray:
    img = img.convert("L")
    if keep_aspect:
        w, h = img.size
        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("L", (target_w, target_h), color=0)
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        canvas.paste(img, (pad_left, pad_top))
        img = canvas
    else:
        img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[..., None]


def _validate_label(text: str, charset: Charset) -> bool:
    if text is None:
        return False
    text = text.strip().upper()
    if not text:
        return False
    for c in text:
        if c not in charset.char_to_idx:
            return False
    return True


def _min_ctc_steps(text: str) -> int:
    if not text:
        return 0
    steps = len(text)
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            steps += 1
    return steps


def load_and_prepare(
    df: pd.DataFrame,
    charset: Charset,
    label_max_length: int,
    target_h: int,
    target_w: int,
    keep_aspect: bool,
    group_column: str | None,
    group_from_path_regex: str | None,
    ctc_max_time_steps: int | None = None,
) -> LoadedSamples:
    image_col = _infer_column(df, IMAGE_CANDIDATES)
    label_col = _infer_column(df, LABEL_CANDIDATES)
    group_col = group_column or _infer_column(df, GROUP_CANDIDATES)

    if image_col is None or label_col is None:
        raise ValueError("Could not infer image or label column names from dataset.")
    if not group_col and not group_from_path_regex:
        print("Warning: no group column found; defaulting to label-based grouping.")

    dataset_root = os.getenv("KAGGLE_DATASET_ROOT", "")

    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    label_lengths: List[int] = []
    texts: List[str] = []
    groups: List[str] = []
    skipped_ctc = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading samples"):
        img = _load_image(row[image_col], dataset_root or None)
        if img is None:
            continue
        text = str(row[label_col]).strip().upper()
        if not _validate_label(text, charset):
            continue
        if ctc_max_time_steps is not None:
            if _min_ctc_steps(text) > ctc_max_time_steps:
                skipped_ctc += 1
                continue
        arr = _preprocess_pil(img, target_h, target_w, keep_aspect)
        encoded, length = encode_label(text, charset, label_max_length)

        images.append(arr)
        labels.append(encoded)
        label_lengths.append(length)
        texts.append(text)

        group_val = None
        if group_col and group_col in row:
            group_val = str(row[group_col])
        if not group_val and isinstance(row[image_col], str):
            group_val = _infer_group_from_path(str(row[image_col]), group_from_path_regex or "")
        if not group_val:
            group_val = text
        groups.append(group_val)

    if not images:
        raise ValueError("No valid samples found after filtering corrupted or invalid rows.")

    if skipped_ctc:
        print(f"Filtered {skipped_ctc} samples exceeding CTC time steps.")

    return LoadedSamples(
        images=np.asarray(images, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        label_lengths=np.asarray(label_lengths, dtype=np.int32),
        texts=texts,
        groups=groups,
    )


def split_by_group(
    samples: LoadedSamples,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> DatasetSplits:
    if not np.isclose(train_ratio + test_ratio + val_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    groups = np.asarray(samples.groups)
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)

    n_total = len(unique_groups)
    n_train = int(round(n_total * train_ratio))
    n_test = int(round(n_total * test_ratio))
    train_groups = set(unique_groups[:n_train].tolist())
    test_groups = set(unique_groups[n_train : n_train + n_test].tolist())
    val_groups = set(unique_groups[n_train + n_test :].tolist())

    def _subset(mask: np.ndarray) -> LoadedSamples:
        return LoadedSamples(
            images=samples.images[mask],
            labels=samples.labels[mask],
            label_lengths=samples.label_lengths[mask],
            texts=[t for t, m in zip(samples.texts, mask) if m],
            groups=[g for g, m in zip(samples.groups, mask) if m],
        )

    mask_train = np.asarray([g in train_groups for g in groups])
    mask_test = np.asarray([g in test_groups for g in groups])
    mask_val = np.asarray([g in val_groups for g in groups])

    return DatasetSplits(
        train=_subset(mask_train),
        test=_subset(mask_test),
        val=_subset(mask_val),
    )


def _motion_blur(image: tf.Tensor) -> tf.Tensor:
    # Predefined small kernels for speed and compatibility.
    kernels = [
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    kernels = [k / np.sum(k) for k in kernels]
    idx = tf.random.uniform([], 0, len(kernels), dtype=tf.int32)
    kernel = tf.constant(np.stack(kernels), dtype=tf.float32)
    kernel = kernel[idx]
    kernel = tf.reshape(kernel, (3, 3, 1, 1))
    image = tf.nn.depthwise_conv2d(image[None, ...], kernel, [1, 1, 1, 1], padding="SAME")
    return image[0]


def _jpeg_artifacts(image: tf.Tensor) -> tf.Tensor:
    img_u8 = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
    img_u8 = tf.image.random_jpeg_quality(img_u8, 30, 90)
    return tf.image.convert_image_dtype(img_u8, tf.float32)


def _low_light(image: tf.Tensor) -> tf.Tensor:
    gamma = tf.random.uniform([], 1.5, 2.5)
    image = tf.image.adjust_gamma(image, gamma=gamma)
    image = tf.clip_by_value(image * 0.7, 0.0, 1.0)
    return image


def augment_image(image: tf.Tensor, cfg: Dict) -> tf.Tensor:
    if tf.random.uniform([]) < float(cfg.get("motion_blur_prob", 0.0)):
        image = _motion_blur(image)
    if tf.random.uniform([]) < float(cfg.get("jpeg_artifact_prob", 0.0)):
        image = _jpeg_artifacts(image)
    if tf.random.uniform([]) < float(cfg.get("low_light_prob", 0.0)):
        image = _low_light(image)
    return image


def make_tf_dataset(
    samples: LoadedSamples,
    batch_size: int,
    augment: bool,
    augment_cfg: Dict,
    shuffle: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (samples.images, samples.labels, samples.label_lengths)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(samples.images))

    def _map(image, label, label_len):
        image = tf.cast(image, tf.float32)
        if augment:
            image = augment_image(image, augment_cfg)
        inputs = {
            "image": image,
            "label_length": tf.expand_dims(label_len, axis=0),
        }
        return inputs, label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
