idf_component_register(SRCS "OCRProject.cpp"
                    INCLUDE_DIRS "."
                    REQUIRES esp32-camera tflite-micro model esp32_s3_eye
                     PRIV_REQUIRES esp_event esp_netif esp_wifi nvs_flash esp_http_server
                    PRIV_REQUIRES esp_timer)
