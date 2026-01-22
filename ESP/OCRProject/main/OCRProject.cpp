#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_camera.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

// LCD (ESP-BSP for ESP32-S3-EYE)
#include "bsp/esp-bsp.h"
#include "bsp/display.h"
#include "esp_lcd_panel_ops.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "ocr_model.h"  // generated from your lightweight_cnn_int8_synth.tflite

// ===================== CONFIG =====================
static const char* TAG = "OCR_MAIN";

// Model input: [1,64,256,1] int8
static constexpr int IN_W = 256;
static constexpr int IN_H = 64;
static constexpr int MAX_TXT = 32;

// Arena
static constexpr int ARENA_SIZE = 320 * 1024;

// Crop (center). Increase H if plate not centered.
static constexpr float CROP_W_FRAC = 0.90f;
static constexpr float CROP_H_FRAC = 0.40f;

// Camera
static constexpr int XCLK_HZ = 10000000;            // 10MHz (stable, avoids noisy frames)
static constexpr framesize_t FRAME_SZ = FRAMESIZE_QVGA; // 320x240
static constexpr pixformat_t PIX_FMT = PIXFORMAT_GRAYSCALE;

// LCD preview
static constexpr bool ENABLE_LCD_PREVIEW = true;
static constexpr int LCD_LINES = 20;

// OCR tweaks
static constexpr bool INVERT_INPUT = false;  // set true if your model expects inverted (try if OCR is empty)
static constexpr int OCR_PERIOD_MS = 300;    // how often to OCR (preview runs faster)

// Wi-Fi AP
static constexpr const char* AP_SSID = "ocr_cam";
static constexpr const char* AP_PASS = "12345678";

// ===================== GLOBALS =====================
// TFLM
static uint8_t* g_arena = nullptr;
static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_in = nullptr;
static TfLiteTensor* g_out = nullptr;
static float g_in_scale = 0.f;
static int g_in_zero = 0;

// Gray frame buffer copy (so we can return fb quickly)
static uint8_t* g_gray = nullptr;
static size_t g_gray_cap = 0;
static int g_w = 0, g_h = 0;

// HTTP
static httpd_handle_t g_http = nullptr;

// LCD
static esp_lcd_panel_handle_t g_panel = nullptr;
static uint16_t* g_lines[2] = {nullptr, nullptr};

// ===================== HELPERS =====================
static inline char idx_to_char(int idx) {
  if (idx >= 0 && idx < 10) return (char)('0' + idx);
  if (idx >= 10 && idx < 36) return (char)('A' + (idx - 10));
  return '\0';
}

static void ensure_buf(uint8_t** p, size_t* cap, size_t need, uint32_t caps) {
  if (*p && *cap >= need) return;
  if (*p) heap_caps_free(*p);
  *p = (uint8_t*)heap_caps_malloc(need, caps);
  if (!*p) *p = (uint8_t*)heap_caps_malloc(need, MALLOC_CAP_8BIT);
  *cap = *p ? need : 0;
}

static inline int8_t quant01(float x01) {
  // x01 in [0..1]
  float scaled = x01 / g_in_scale + (float)g_in_zero;
  int32_t q = (int32_t)lroundf(scaled);
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

// Copy grayscale fb -> g_gray and return fb ASAP
static bool capture_gray_frame() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return false;

  if (fb->format != PIXFORMAT_GRAYSCALE) {
    ESP_LOGW(TAG, "Unexpected format %d", fb->format);
    esp_camera_fb_return(fb);
    return false;
  }

  g_w = fb->width;
  g_h = fb->height;
  const size_t need = (size_t)g_w * (size_t)g_h;

  ensure_buf(&g_gray, &g_gray_cap, need, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!g_gray) {
    esp_camera_fb_return(fb);
    return false;
  }

  memcpy(g_gray, fb->buf, need);
  esp_camera_fb_return(fb);
  return true;
}

// Preprocess: crop center band -> keep_aspect resize + letterbox
static void preprocess_to_model(const uint8_t* src, int sw, int sh, int8_t* dst) {
  const int cw = std::max(1, std::min((int)lroundf(sw * CROP_W_FRAC), sw));
  const int ch = std::max(1, std::min((int)lroundf(sh * CROP_H_FRAC), sh));
  const int cx = (sw - cw) / 2;
  const int cy = (sh - ch) / 2;

  const float scale_w = (float)IN_W / (float)cw;
  const float scale_h = (float)IN_H / (float)ch;
  const float s = std::min(scale_w, scale_h);

  const int rw = std::max(1, (int)lroundf(cw * s));
  const int rh = std::max(1, (int)lroundf(ch * s));
  const int pad_x = (IN_W - rw) / 2;
  const int pad_y = (IN_H - rh) / 2;

  for (int y = 0; y < IN_H; y++) {
    for (int x = 0; x < IN_W; x++) {
      uint8_t p = 0;
      if (x >= pad_x && x < pad_x + rw && y >= pad_y && y < pad_y + rh) {
        int sx = cx + (int)((x - pad_x) / s);
        int sy = cy + (int)((y - pad_y) / s);
        sx = std::min(std::max(sx, 0), sw - 1);
        sy = std::min(std::max(sy, 0), sh - 1);
        p = src[sy * sw + sx];
      }
      if (INVERT_INPUT) p = 255 - p;
      const float x01 = (float)p / 255.0f;
      dst[y * IN_W + x] = quant01(x01);
    }
  }
}

// Greedy CTC decode assuming blank is last class (C-1)
// If OCR is always empty, try blank=0.
static void ctc_decode(const int8_t* logits, int T, int C, int blank, char* out, int out_len) {
  out[0] = '\0';
  int out_i = 0;
  int prev = -1;

  const float os = g_out->params.scale;
  const int oz = g_out->params.zero_point;

  for (int t = 0; t < T; t++) {
    int best = 0;
    float bestv = (logits[t * C + 0] - oz) * os;
    for (int c = 1; c < C; c++) {
      float v = (logits[t * C + c] - oz) * os;
      if (v > bestv) {
        bestv = v;
        best = c;
      }
    }

    if (best == blank) { prev = best; continue; }
    if (best == prev) continue;

    char ch = idx_to_char(best);
    if (ch && out_i + 1 < out_len) {
      out[out_i++] = ch;
      out[out_i] = '\0';
    }
    prev = best;
  }
}

// Simple brightness stats to see if image is really black
static void log_gray_stats(const uint8_t* gray, int w, int h) {
  int mn = 255, mx = 0;
  int64_t sum = 0;
  int samples = 0;
  const int step = 50; // sample every 50 pixels
  const int total = w * h;

  for (int i = 0; i < total; i += step) {
    int v = gray[i];
    mn = std::min(mn, v);
    mx = std::max(mx, v);
    sum += v;
    samples++;
  }
  int avg = samples ? (int)(sum / samples) : 0;
  ESP_LOGI(TAG, "GRAY stats: min=%d max=%d avg~=%d", mn, mx, avg);
}

// ===================== LCD PREVIEW =====================
static bool init_lcd() {
  if (!ENABLE_LCD_PREVIEW) return false;

  bsp_display_config_t disp_cfg = {
      .max_transfer_sz = BSP_LCD_H_RES * LCD_LINES * sizeof(uint16_t),
  };
  esp_lcd_panel_io_handle_t io = nullptr;
  esp_err_t err = bsp_display_new(&disp_cfg, &g_panel, &io);
  if (err != ESP_OK || !g_panel) {
    ESP_LOGW(TAG, "LCD init failed (0x%x). LCD preview disabled.", err);
    g_panel = nullptr;
    return false;
  }

  esp_lcd_panel_disp_on_off(g_panel, true);
  bsp_display_backlight_on();

  for (int i = 0; i < 2; i++) {
    g_lines[i] = (uint16_t*)heap_caps_malloc(BSP_LCD_H_RES * LCD_LINES * sizeof(uint16_t),
                                             MALLOC_CAP_DMA);
    if (!g_lines[i]) {
      ESP_LOGW(TAG, "LCD line buffer alloc failed");
      g_panel = nullptr;
      return false;
    }
  }

  ESP_LOGI(TAG, "LCD ready: %dx%d", BSP_LCD_H_RES, BSP_LCD_V_RES);
  return true;
}

static void draw_target_box(uint16_t* buf, int w, int start_y, int lines) {
  // Draw a simple green rectangle indicating crop area (approx)
  const int box_w = (int)(w * CROP_W_FRAC);
  const int box_h = (int)(BSP_LCD_V_RES * CROP_H_FRAC);
  const int box_x = (w - box_w) / 2;
  const int box_y = (BSP_LCD_V_RES - box_h) / 2;

  const uint16_t green = 0x07E0;

  for (int dy = 0; dy < lines; dy++) {
    int y = start_y + dy;
    if (y < box_y || y >= box_y + box_h) continue;

    if (y == box_y || y == box_y + box_h - 1) {
      for (int x = box_x; x < box_x + box_w; x++) buf[dy * w + x] = green;
    } else {
      buf[dy * w + box_x] = green;
      buf[dy * w + (box_x + box_w - 1)] = green;
    }
  }
}

static void lcd_preview_gray(const uint8_t* gray, int sw, int sh) {
  if (!g_panel || !gray) return;

  const int dw = BSP_LCD_H_RES;
  const int dh = BSP_LCD_V_RES;
  int line_idx = 0;

  for (int y = 0; y < dh; y += LCD_LINES) {
    const int lines = std::min(LCD_LINES, dh - y);
    uint16_t* out = g_lines[line_idx];

    for (int dy = 0; dy < lines; ++dy) {
      const int sy = (y + dy) * sh / dh;
      const int srow = sy * sw;
      for (int x = 0; x < dw; ++x) {
        const int sx = x * sw / dw;
        uint8_t p = gray[srow + sx];
        // Gray -> RGB565
        uint16_t rgb = (uint16_t)(((p >> 3) << 11) | ((p >> 2) << 5) | (p >> 3));
        out[dy * dw + x] = rgb;
      }
    }

    draw_target_box(out, dw, y, lines);
    esp_lcd_panel_draw_bitmap(g_panel, 0, y, dw, y + lines, out);
    line_idx ^= 1;
  }
}

// ===================== CAMERA INIT =====================
static bool init_camera() {
  camera_config_t cfg = {};
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer = LEDC_TIMER_0;

  // ESP32-S3-EYE pin map (your working map)
  cfg.pin_d0 = 11; cfg.pin_d1 = 9;  cfg.pin_d2 = 8;  cfg.pin_d3 = 10;
  cfg.pin_d4 = 12; cfg.pin_d5 = 18; cfg.pin_d6 = 17; cfg.pin_d7 = 16;
  cfg.pin_xclk = 15; cfg.pin_pclk = 13; cfg.pin_vsync = 6; cfg.pin_href = 7;
  cfg.pin_sccb_sda = 4; cfg.pin_sccb_scl = 5;
  cfg.pin_pwdn = -1;
  cfg.pin_reset = -1;

  cfg.xclk_freq_hz = XCLK_HZ;
  cfg.pixel_format = PIX_FMT;
  cfg.frame_size = FRAME_SZ;

  cfg.jpeg_quality = 12;
  cfg.fb_count = 2;
  cfg.fb_location = CAMERA_FB_IN_PSRAM;
  cfg.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
    return false;
  }

  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    // Brighten the image (helps phone screens)
    s->set_brightness(s, 2);     // -2..2
    s->set_contrast(s, 2);       // 0..2
    s->set_saturation(s, 0);

    s->set_whitebal(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);

    s->set_aec2(s, 1);
    s->set_ae_level(s, 2);
    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);
  }

  // Throw away first frames (stabilize exposure)
  for (int i = 0; i < 6; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    vTaskDelay(pdMS_TO_TICKS(30));
  }

  ESP_LOGI(TAG, "Camera initialized: GRAYSCALE %dx%d, fb_count=2, xclk=%d",
           320, 240, XCLK_HZ);
  return true;
}

// ===================== WIFI + HTTP =====================
static esp_err_t gray_handler(httpd_req_t* req) {
  if (!g_gray || g_w <= 0 || g_h <= 0) {
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "No frame yet");
    return ESP_OK;
  }

  char header[64];
  int n = snprintf(header, sizeof(header), "P5\n%d %d\n255\n", g_w, g_h);

  httpd_resp_set_type(req, "application/octet-stream");
  httpd_resp_send_chunk(req, header, n);
  httpd_resp_send_chunk(req, (const char*)g_gray, (size_t)g_w * (size_t)g_h);
  httpd_resp_send_chunk(req, nullptr, 0);
  return ESP_OK;
}

static void start_http() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = 80;

  if (httpd_start(&g_http, &cfg) != ESP_OK) {
    ESP_LOGE(TAG, "HTTP server start failed");
    g_http = nullptr;
    return;
  }

  httpd_uri_t uri = {};
  uri.uri = "/gray.pgm";
  uri.method = HTTP_GET;
  uri.handler = gray_handler;
  httpd_register_uri_handler(g_http, &uri);

  ESP_LOGI(TAG, "HTTP endpoint ready: /gray.pgm");
}

static void wifi_ap_start() {
  ESP_ERROR_CHECK(nvs_flash_init());
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_ap();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));

  wifi_config_t ap = {};
  std::strncpy((char*)ap.ap.ssid, AP_SSID, sizeof(ap.ap.ssid));
  std::strncpy((char*)ap.ap.password, AP_PASS, sizeof(ap.ap.password));
  ap.ap.ssid_len = std::strlen(AP_SSID);
  ap.ap.max_connection = 4;
  ap.ap.authmode = WIFI_AUTH_WPA_WPA2_PSK;
  ap.ap.channel = 1;

  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
  ESP_ERROR_CHECK(esp_wifi_start());

  ESP_LOGI(TAG, "WiFi AP started: SSID=%s PASS=%s", AP_SSID, AP_PASS);
  ESP_LOGI(TAG, "Open: http://192.168.4.1/gray.pgm");
}

// ===================== TFLM INIT =====================
static bool init_tflm() {
  g_model = tflite::GetModel(ocr_model_data);
  if (!g_model || g_model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema mismatch");
    return false;
  }

  g_arena = (uint8_t*)heap_caps_malloc(ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!g_arena) g_arena = (uint8_t*)heap_caps_malloc(ARENA_SIZE, MALLOC_CAP_8BIT);
  if (!g_arena) {
    ESP_LOGE(TAG, "Arena alloc failed");
    return false;
  }

  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddAdd();
  resolver.AddConcatenation();
  resolver.AddGather();
  resolver.AddPack();
  resolver.AddShape();

  static tflite::MicroInterpreter interp(g_model, resolver, g_arena, ARENA_SIZE);
  g_interpreter = &interp;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors failed");
    return false;
  }

  g_in = g_interpreter->input(0);
  g_out = g_interpreter->output(0);

  if (!g_in || g_in->type != kTfLiteInt8) {
    ESP_LOGE(TAG, "Expected int8 input");
    return false;
  }
  if (!g_out || g_out->type != kTfLiteInt8) {
    ESP_LOGE(TAG, "Expected int8 output");
    return false;
  }

  g_in_scale = g_in->params.scale;
  g_in_zero  = g_in->params.zero_point;

  // Output dims [1,T,C]
  int T = g_out->dims->data[g_out->dims->size - 2];
  int C = g_out->dims->data[g_out->dims->size - 1];

  ESP_LOGI(TAG, "Model loaded: %u bytes", ocr_model_data_len);
  ESP_LOGI(TAG, "Input shape: 1x%dx%dx1", IN_H, IN_W);
  ESP_LOGI(TAG, "Output shape: 1x%dx%d", T, C);
  ESP_LOGI(TAG, "Input quant: scale=%.6f zero=%d", g_in_scale, g_in_zero);
  return true;
}

// ===================== APP MAIN =====================
extern "C" void app_main() {
  ESP_LOGI(TAG, "Starting OCR + LCD preview + Web debug");

  if (!init_camera()) {
    ESP_LOGE(TAG, "Camera setup failed");
    return;
  }

  // LCD preview (so screen is NOT black)
  init_lcd();

  // Start AP + HTTP so you can download frames
  wifi_ap_start();
  start_http();

  if (!init_tflm()) {
    ESP_LOGE(TAG, "TFLM setup failed");
    return;
  }

  int64_t last_ocr = 0;

  while (true) {
    if (!capture_gray_frame()) {
      ESP_LOGW(TAG, "Capture failed");
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    // Show on LCD always
    if (ENABLE_LCD_PREVIEW && g_panel) {
      lcd_preview_gray(g_gray, g_w, g_h);
    }

    // Log stats occasionally (helps diagnose "black")
    static int frame = 0;
    frame++;
    if ((frame % 20) == 0) {
      log_gray_stats(g_gray, g_w, g_h);
    }

    // Run OCR periodically
    int64_t now = esp_timer_get_time();
    if ((now - last_ocr) >= (int64_t)OCR_PERIOD_MS * 1000) {
      last_ocr = now;

      preprocess_to_model(g_gray, g_w, g_h, g_in->data.int8);

      int64_t t0 = esp_timer_get_time();
      if (g_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
      } else {
        int64_t t1 = esp_timer_get_time();

        int T = g_out->dims->data[g_out->dims->size - 2];
        int C = g_out->dims->data[g_out->dims->size - 1];

        char text[MAX_TXT] = {0};

        // Most models: blank is last class (C-1)
        int blank = C - 1;
        ctc_decode(g_out->data.int8, T, C, blank, text, sizeof(text));

        ESP_LOGI(TAG, "OCR='%s' (%lld ms)", text, (long long)((t1 - t0) / 1000));
      }
    }

    vTaskDelay(pdMS_TO_TICKS(30));
  }
}
