#ifndef OCR_MODEL_H
#define OCR_MODEL_H

// OCR Model for ESP32-S3-EYE (lightweight_cnn INT8 PTQ)
// Input: 1 x 64 x 256 x 1 grayscale, normalized to [0,1] then int8 quantized.
// Output: 1 x T x 37 (CTC logits), charset "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" + blank.

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char ocr_model_data[];
extern const unsigned int ocr_model_data_len;

#ifdef __cplusplus
}
#endif

#endif
