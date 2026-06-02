#include "esp_log.h"
#include "esp_littlefs.h"

static const char *FS_TAG = "FILESYS";

void init_fs() {
	ESP_LOGI(FS_TAG, "Initalizing littlefs!");
	esp_vfs_littlefs_conf_t conf = {
        .base_path = "/model_data",
        .partition_label = "model",       
		.format_if_mount_failed = false,
        .dont_mount = false
    };
	esp_err_t ret = esp_vfs_littlefs_register(&conf);
	if (ret != ESP_OK) {
		if (ret == ESP_FAIL) {
			ESP_LOGI(FS_TAG, "Could not mount filesystem!");
		} else if (ret == ESP_ERR_NOT_FOUND) {
            ESP_LOGE(FS_TAG, "Failed to find LittleFS partition labeled 'model'");
        } else {
            ESP_LOGE(FS_TAG, "Failed to initialize LittleFS (%s)", esp_err_to_name(ret));
        }
	}
}

FILE* get_model() {
	FILE *f = fopen(
		"/model_data/model.tflite",
		"r"
	);

	if (f == NULL) {
		ESP_LOGI(FS_TAG, "Could not find model.tflite. Was it flashed correctly? Did you call init_fs?");
		return NULL;
	}

	return f;
}
