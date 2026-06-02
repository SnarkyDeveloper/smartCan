cmake_minimum_required(VERSION 3.16)

idf_component_register(
    SRCS 
        "main.c"
        "components/tensor/tensor.cpp"
		"components/camera.c"
    INCLUDE_DIRS 
        "."
        "components"
        "components/tensor"
    REQUIRES 
        esp32-camera 
        esp-tflite-micro 
        esp-nn 
 		esp_jpeg
		esp_driver_gpio
    PRIV_REQUIRES 
		esp_mm
		esp_driver_ledc
)
