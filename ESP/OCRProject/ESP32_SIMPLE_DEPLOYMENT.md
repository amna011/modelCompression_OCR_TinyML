# ESP32-S3-EYE OCR Deployment (ESP-IDF)

This guide deploys the winner model (lightweight_cnn INT8 PTQ) on ESP32-S3-EYE.

## 1) Prerequisites

- ESP-IDF v5.x installed and configured for `esp32s3`.
- `esp32-camera` and `esp-tflite-micro` components available.
- Python environment with access to this repo.

## 2) Convert the winner model to C

From the repo root:

```bash
.\.venv\Scripts\python scripts\tflite_to_cc.py --in models\lightweight_cnn_int8.tflite --out ESP\OCRProject\components\model\ocr_model.c --var ocr_model_data
```

This writes the C array used by `ESP/OCRProject/components/model/ocr_model.h`.

## 3) Build and flash

```bash
cd ESP\OCRProject
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py flash monitor
```

Suggested menuconfig settings:
- Enable PSRAM (ESP32S3-specific).
- Enable camera support.

## 4) Live test checklist

- Serial logs show:
  - Model loaded and tensor arena allocated.
  - Input shape `1 x 64 x 256 x 1`.
  - Output shape `1 x T x 37`.
  - Inference time per frame.
- Point the camera at a license plate; verify decoded strings are stable.

## 5) Troubleshooting

### AllocateTensors fails
- Increase `kTensorArenaSize` in `ESP/OCRProject/main/OCRProject.cpp`.
- Ensure the model uses supported TFLM ops. If you see missing op errors
  (e.g., `REDUCE_PROD`), re-export the model with static reshape ops.

### Camera init fails
- Reduce `xclk_freq_hz` to 10 MHz.
- Fall back to `FRAMESIZE_QQVGA` (already attempted in code).

### Output is blank
- Improve lighting and contrast.
- Confirm preprocessing matches training (64x256, aspect preserved, int8 quantize).

## 6) Expected serial output

```
I (XXX) OCR_MAIN: Starting OCR on ESP32-S3-EYE
I (XXX) OCR_MAIN: Camera initialized (QVGA grayscale).
I (XXX) OCR_MAIN: Model loaded: 64200 bytes
I (XXX) OCR_MAIN: Input shape: 1 x 64 x 256 x 1
I (XXX) OCR_MAIN: Output shape: 1 x 8 x 37
I (XXX) OCR_MAIN: Inference time: 10 ms
I (XXX) OCR_MAIN: OCR Result: ABC123
```
