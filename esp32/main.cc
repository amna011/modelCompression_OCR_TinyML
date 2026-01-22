#include <cstdint>
#include <cstring>

#include "model_data.cc"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr int kTensorArenaSize = 300 * 1024;
static uint8_t g_tensor_arena[kTensorArenaSize];
}

static void GreedyCTCDecode(const int8_t* logits, int time_steps, int num_classes,
                            int blank_index, char* out, int out_len) {
  int out_idx = 0;
  int prev = -1;
  for (int t = 0; t < time_steps; ++t) {
    int best = 0;
    int best_val = logits[t * num_classes];
    for (int c = 1; c < num_classes; ++c) {
      int val = logits[t * num_classes + c];
      if (val > best_val) {
        best_val = val;
        best = c;
      }
    }
    if (best == prev || best == blank_index) {
      prev = best;
      continue;
    }
    if (out_idx + 1 < out_len) {
      out[out_idx++] = static_cast<char>(best);
      out[out_idx] = '\0';
    }
    prev = best;
  }
}

int main() {
  const tflite::Model* model = tflite::GetModel(g_ocr_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }

  tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();

  tflite::MicroInterpreter interpreter(
      model, resolver, g_tensor_arena, kTensorArenaSize);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 2;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  // TODO: capture camera frame, preprocess to 64x256 grayscale, normalize, quantize to int8.
  // Fill input->data.int8 with the quantized image.

  if (interpreter.Invoke() != kTfLiteOk) {
    return 3;
  }

  // TODO: map class index to actual characters. This placeholder uses raw index.
  char decoded[32] = {0};
  GreedyCTCDecode(output->data.int8, output->dims->data[1], output->dims->data[2],
                  output->dims->data[2] - 1, decoded, sizeof(decoded));

  // TODO: print decoded via serial.
  (void)decoded;

  return 0;
}
