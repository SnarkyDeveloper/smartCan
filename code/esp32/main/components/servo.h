#include "freertos/idf_additions.h"
#include "iot_servo.h"
#include "esp_log.h"
#include "hal/ledc_types.h"
#include "tensor.h"

#define SERVO_PIN_PRIMARY 21

static const char *SERVO_TAG = "SERVO";

bool init_servos() {
	servo_config_t sc = {
		.max_angle = 180,
		.min_width_us = 500,
		.max_width_us = 2500,
		.freq = 50,
		.timer_number = LEDC_TIMER_0,
		.channel_number = 1, // only initialize one channel servo for now
		.channels = {
			.servo_pin = {
				SERVO_PIN_PRIMARY	
			},
			.ch = {
				LEDC_CHANNEL_0
			}
		}
	};

	esp_err_t servo = iot_servo_init(LEDC_LOW_SPEED_MODE, &sc);
	if (servo != ESP_OK) {
		ESP_LOGE(SERVO_TAG, "Failed to initialize servo: %s", esp_err_to_name(servo));
		return false;
	}

	return true;
}


#define RECYCLABLE_ANGLE 180
#define ORGANIC_ANGLE 5
#define FLAT 93

void move_servos(Identification result) {
	if (result.confidence < 0.2) {
		ESP_LOGI(SERVO_TAG, "Low confidence (%.2f%%). Not moving servos. Most likely nothing in trash", result.confidence * 100.0);
		return;
	}
	float angle = FLAT; // default to flat
	if (result.identification == 1) {
		angle = RECYCLABLE_ANGLE;
	} else if (result.identification == 0) {
		angle = ORGANIC_ANGLE;
	}
	vTaskDelay(pdMS_TO_TICKS(2000)); // delay for 2s so users dont hit their hands
	iot_servo_write_angle(
		LEDC_LOW_SPEED_MODE,
		0,
		angle
	);
	
	vTaskDelay(pdMS_TO_TICKS(3000)); // keep the servo open for 5s before closing it again
	iot_servo_write_angle(
		LEDC_LOW_SPEED_MODE,
		0,
		FLAT
	);
}
