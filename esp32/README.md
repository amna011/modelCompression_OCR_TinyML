# ESP32-S3-EYE Deployment

This folder contains deployment notes for running the winning INT8 TFLite model on ESP32-S3-EYE using TensorFlow Lite Micro.

## Steps

1) Convert the winner TFLite file to a C array:

```bash
python scripts/tflite_to_cc.py --in models/lightweight_cnn_int8.tflite --out esp32/model_data.cc --var g_ocr_model
```

2) Integrate the model into your ESP-IDF or Arduino TFLM project:

- Add `model_data.cc` to the project.
- Ensure `g_ocr_model` and `g_ocr_model_len` are linked.

3) Preprocessing on device must match training:

- Convert camera RGB to grayscale.
- Resize to 64x256 with aspect ratio preserved and padded.
- Normalize to [0, 1] and then quantize to int8 using input scale/zero-point.

4) CTC decoding:

- Run argmax across the class dimension for each time step.
- Remove repeated characters and blanks.

5) Validation:

- Compare decoded string vs. PC TFLite output for the same frames.
- Confirm stable FPS and no watchdog resets.

## Notes

- A full ESP-IDF reference project is available in `ESP/OCRProject`.

- Confirm input/output tensor quantization params via `interpreter.get_input_details()`.
- Do not change preprocessing between PC and ESP32.
