#include "esp_camera.h" 
#include "esp_log.h"
#include "components/fs.h"
#include "components/tensor/tensor.h"
#include "components/camera.h"
#include "components/servo.h"
#include "freertos/FreeRTOS.h"

static const char *TAG = "LOADER";
static const char *LOOP_TAG = "[LOOP] ";

#define LOOP_TIME 3

void app_main(void) {
	vTaskDelay(pdMS_TO_TICKS(500)); 
	init_fs();
	FILE* modelfile = get_model();
	if (modelfile == NULL) {
		ESP_LOGI(TAG, "Model file could not be loaded. Exiting.");
		return;
	}
	uint8_t* buffer = nullptr;
	load_model_from_file(modelfile, &buffer);
	fclose(modelfile);
	if (buffer == nullptr) {
		ESP_LOGI(TAG, "Model loading failed. Exiting.");
		return;
	}

	if (initalize_camera() == false) {
		ESP_LOGI(TAG, "Camera initialization failed. Exiting.");
		return;
	}

	if (init_servos() == false) {
		ESP_LOGI(TAG, "Servo initialization failed. Exiting.");
		return;
	}

	int fails = 0;
	ESP_LOGI(TAG, "Initialization complete. Starting inference loop.");
	while (1) {
		if (fails > 5) { // doesnt matter since itll immediately rerun loop if fails because of continue
			ESP_LOGI(TAG, "Too many camera capture failures. Exiting.");
			break;
		}
		camera_fb_t* fb = get_frame();
		if (fb == nullptr) {
			ESP_LOGE(TAG, "Failed to capture image from camera.");
			fails++;
			continue;
		}

		Identification result = run_inference((void*)buffer, (void*)fb);
		ESP_LOGI(LOOP_TAG, "Inference result: %s with confidence %.2f%%", 
					result.identification == 1 ? "Recyclable" : "Organic", 
					result.confidence * 100.0);
		move_servos(result);

		esp_camera_fb_return(fb); // cleanup camera frame buffer after inference
		
		vTaskDelay(pdMS_TO_TICKS(LOOP_TIME * 1000));
	}
}

// #include <stdio.h>
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "esp_log.h"
// #include "esp_camera.h"
// #include "camera.h"
//
// static const char *TAG = "DUMP";
//
// void app_main(void)
// {
//     ESP_LOGI(TAG, "Initializing camera...");
//
//     if (!initalize_camera()) {
//         ESP_LOGE(TAG, "Camera init failed");
//         return;
//     }
//
//     ESP_LOGI(TAG, "Capturing single frame...");
//
//     camera_fb_t *fb = esp_camera_fb_get();
//
//     if (!fb) {
//         ESP_LOGE(TAG, "Capture failed");
//         return;
//     }
//
//     ESP_LOGI(TAG,
//              "Captured frame: %ux%u len=%u format=%d",
//              fb->width,
//              fb->height,
//              (unsigned)fb->len,
//              fb->format);
//
//     printf("\n=== FRAME START ===\n");
//
//     for (size_t i = 0; i < fb->len; i++) {
//         printf("%02X", fb->buf[i]);
//
//         if ((i & 15) == 15)
//             printf("\n");
//         else
//             printf(" ");
//     }
//
//     printf("\n=== FRAME END ===\n");
//
//     esp_camera_fb_return(fb);
//
//     ESP_LOGI(TAG, "Done. Halting.");
//
//     while (1) {
//         vTaskDelay(portMAX_DELAY);
//     }
// }
