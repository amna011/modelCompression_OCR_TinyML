# OCR on ESP32-S3-EYE

This project implements a full OCR pipeline for license plate recognition on ESP32-class microcontrollers using TensorFlow Lite Micro. It follows the phase and dataset requirements from the task description.

## Quick start

1) Create a Python environment and install dependencies:

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
```

2) Train FP32 models:

```bash
python -m ocr.train --config configs/default.yaml --model lightweight_cnn
python -m ocr.train --config configs/default.yaml --model mobilenet_v2
python -m ocr.train --config configs/default.yaml --model efficientnet_b0
```

3) Run evaluation:

```bash
python -m ocr.evaluate --config configs/default.yaml --model_path models/lightweight_cnn_fp32.keras
```

4) Quantize:

```bash
python -m ocr.compress --config configs/default.yaml --model_path models/lightweight_cnn_fp32.keras --out models/lightweight_cnn_int8.tflite
```

QAT option:

```bash
python -m ocr.compress --config configs/default.yaml --model_path models/lightweight_cnn_fp32.keras --out models/lightweight_cnn_int8.tflite --qat --fine_tune_epochs 10
```

Structured pruning (lightweight_cnn only):

```bash
python -m ocr.prune --model_path models/lightweight_cnn_fp32.keras --out models/lightweight_cnn_pruned.keras --prune_ratio 0.3 --min_filters 8 --use_relu6
```

Knowledge distillation:

```bash
python -m ocr.distill --config configs/default.yaml --teacher models/efficientnet_b0_fp32.keras --student lightweight_cnn --out models/lightweight_cnn_distilled.keras
```

5) Benchmark and compare:

```bash
python -m ocr.benchmark --results benchmarks/results.json
```

6) ESP32 deployment steps are in `esp32/README.md`.

## Repository layout

- `src/ocr`: data pipeline, models, training, evaluation, compression
- `configs`: experiment configs
- `models`: saved checkpoints and TFLite files
- `benchmarks`: consolidated results table and logs
- `esp32`: deployment notes and code skeleton

## Notes

- All images are grayscale, resized to 64x256 with aspect ratio preserved and padded, normalized to [0, 1].
- CTC is used end-to-end. No character segmentation.
- CER and exact match are the required metrics.
- Benchmark chart and winner rationale are in `docs/benchmark_chart.md`.
- Winner before/after chart is in `docs/figures/winner_before_after.svg`.
- All-models before/after chart is in `docs/figures/all_models_before_after.svg`.
- Paper-ready figures can be generated with `scripts/generate_paper_figures.py`.
- If Kaggle download is blocked, set `KAGGLE_DATASET_ROOT` to a local copy of the dataset and optionally
  `KAGGLE_DATASET_FILE` to the metadata file (CSV/Parquet/etc.).
- If training logs show `No valid path found` from CTC, set `ctc.max_time_steps` to the minimum time steps
  supported by the model outputs to filter incompatible samples consistently.
- MobileNetV2/EfficientNet can use intermediate feature layers (see config) to preserve more time steps.
