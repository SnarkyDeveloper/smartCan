#include "camera.h"
#include "esp_log.h"
#include "sensor.h"
#include "driver/gpio.h"       
#include "freertos/FreeRTOS.h" 
#include "freertos/task.h"

static const char *CAM_TAG = "CAMERA";
static uint8_t *rgb_conversion_buffer = NULL;

bool initalize_camera(void) {
    gpio_reset_pin(SIOD_GPIO_NUM);
    gpio_reset_pin(SIOC_GPIO_NUM);
    
    gpio_set_direction(SIOD_GPIO_NUM, GPIO_MODE_OUTPUT);
    gpio_set_direction(SIOC_GPIO_NUM, GPIO_MODE_OUTPUT);
    
    gpio_set_level(SIOD_GPIO_NUM, 0);
    gpio_set_level(SIOC_GPIO_NUM, 0);
    vTaskDelay(pdMS_TO_TICKS(100)); 

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;

    config.xclk_freq_hz = 10000000; // Stable 10MHz XCLK
    
    // Initialize physically in Grayscale 96x96 to bypass ll_cam hardware limitations
    config.pixel_format = PIXFORMAT_RGB565;  
    config.frame_size = FRAMESIZE_96X96;  
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.sccb_i2c_port = 0; 
    esp_err_t Err = esp_camera_init(&config);
    if (Err != ESP_OK) {
        ESP_LOGE(CAM_TAG, "Camera initialization failed with error 0x%x", Err);
        return false;
    }
	sensor_t* s = esp_camera_sensor_get();
	if (s != NULL) {
		s->set_gainceiling(s, GAINCEILING_16X);
		s->set_brightness(s, 1); // amplify
		s->set_contrast(s, 1);

		s->set_whitebal(s, 1);
		s->set_exposure_ctrl(s, 1);
		
	}
    // Allocate persistent conversion pool: 96 * 96 * 3 = 27648 bytes
    rgb_conversion_buffer = (uint8_t *)heap_caps_malloc(96 * 96 * 3, MALLOC_CAP_SPIRAM);
    if (rgb_conversion_buffer == NULL) {
        ESP_LOGE(CAM_TAG, "Failed to allocate memory array for RGB conversion workspace.");
        return false;
    }

    ESP_LOGI(CAM_TAG, "Camera hardware successfully verified and initialized.");
    return true;
}

camera_fb_t* get_frame(void) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb == NULL) {
        ESP_LOGE(CAM_TAG, "Failed to capture raw frame from camera sensor.");
        return NULL;
    }

    // Process conversion directly inside the fetched frame container bounds
    size_t pixel_count = fb->width * fb->height;
    
    // Copy backwards into intermediate buffer to prevent clobbering raw index fields
    for (int i = (int)pixel_count - 1; i >= 0; i--) {
        uint8_t gray_val = fb->buf[i];
        rgb_conversion_buffer[i * 3 + 0] = gray_val; // R
        rgb_conversion_buffer[i * 3 + 1] = gray_val; // G
        rgb_conversion_buffer[i * 3 + 2] = gray_val; // B
    }

    // Hot-swap the underlying pointer and properties to fake raw RGB888 structure
    fb->buf = rgb_conversion_buffer;
    fb->len = pixel_count * 3;
    fb->format = PIXFORMAT_RGB888; // Forces this property explicitly to integer value 5

    return fb;
}
