#include "tensor.h"
#include "esp_log.h"
#include "esp_camera.h"
#include "esp_heap_caps.h" 
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_cache.h" 

constexpr int tensor_arena_size = 1000 * 1024; // 1mb/8mb
static uint8_t* tensor_arena = nullptr; 
static const char *TF_TAG = "TFLite"; 

static tflite::MicroInterpreter* global_interpreter = nullptr;
static tflite::MicroMutableOpResolver<16> op_resolver;
static bool ops_registered = false;
static bool interpreter_broken = false; 

extern "C" void* load_model_from_file(FILE* file_ptr, uint8_t** model_buffer_out) {
    if (file_ptr == NULL || model_buffer_out == NULL) return NULL;

    fseek(file_ptr, 0, SEEK_END);
    long model_size = ftell(file_ptr);
    fseek(file_ptr, 0, SEEK_SET);

    if (model_size <= 0) {
        ESP_LOGE(TF_TAG, "Model file size reports as invalid or 0 bytes.");
        return NULL;
    }

    size_t aligned_model_size = (model_size + 63) & ~63;

    uint8_t* buffer = (uint8_t*)heap_caps_aligned_alloc(64, aligned_model_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (buffer == NULL) {
        ESP_LOGE(TF_TAG, "Failed to allocate memory buffer in PSRAM.");
        return NULL;
    }

    size_t bytes_read = fread(buffer, 1, model_size, file_ptr);
    ESP_LOGI(TF_TAG, "Read %d out of %ld expected model bytes from storage.", bytes_read, model_size);
    
    if (bytes_read != model_size) {
        ESP_LOGE(TF_TAG, "File read mismatch error! Storage data might be corrupted.");
        heap_caps_free(buffer);
        return NULL;
    }

    if (aligned_model_size > model_size) {
        memset(buffer + model_size, 0, aligned_model_size - model_size);
    }

    esp_err_t cache_err = esp_cache_msync(buffer, aligned_model_size, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
    if (cache_err != ESP_OK) {
        ESP_LOGE(TF_TAG, "Cache synchronization failed: 0x%x", cache_err);
        heap_caps_free(buffer);
        return NULL;
    }

    *model_buffer_out = buffer;
    ESP_LOGI(TF_TAG, "Model file safely buffered and synchronized.");
    return (void*)buffer; 
}

extern "C" Identification run_inference(void* model_buffer_ptr, void* camera_fb_ptr) {
    Identification result = {0.0, 0}; 

    if (model_buffer_ptr == nullptr || camera_fb_ptr == nullptr || interpreter_broken) {
        return result;
    }

    if (tensor_arena == nullptr) {
        tensor_arena = (uint8_t*)heap_caps_aligned_alloc(64, tensor_arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (tensor_arena == nullptr) return result;
    }

    if (global_interpreter == nullptr) {
        const tflite::Model* model = tflite::GetModel((uint8_t*)model_buffer_ptr);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            ESP_LOGE(TF_TAG, "Model flatbuffer schema validation failed!");
            return result;
        }

        if (!ops_registered) {
            op_resolver.AddConv2D();
            op_resolver.AddDepthwiseConv2D();
            op_resolver.AddReshape();
            op_resolver.AddSoftmax();       
            op_resolver.AddAdd();           
            op_resolver.AddMul();           
            op_resolver.AddAveragePool2D(); 
            op_resolver.AddPad();           
            op_resolver.AddPadV2();         
            op_resolver.AddMean();          
            op_resolver.AddQuantize();      
            op_resolver.AddDequantize();    
            op_resolver.AddLogistic();      
            
            op_resolver.AddFullyConnected(); 
            
            ops_registered = true;
        }

        global_interpreter = new tflite::MicroInterpreter(model, op_resolver, tensor_arena, tensor_arena_size);
        
        if (global_interpreter->AllocateTensors() != kTfLiteOk) {
            ESP_LOGE(TF_TAG, "Tensor allocation failed.");
            interpreter_broken = true; 
            return result;
        }
    }
	
	camera_fb_t* fb = (camera_fb_t*)camera_fb_ptr;
    TfLiteTensor* input = global_interpreter->input(0);
	
	if (input->type == kTfLiteInt8) {
        int8_t* input_buffer = input->data.int8;
        float scale = input->params.scale;
        int32_t zero_point = input->params.zero_point;

        static int debug_pixel_waves = 0;
        if (debug_pixel_waves > 100) debug_pixel_waves = 0; 

        for (size_t i = 0; i < fb->len; ++i) {
            float normalized = ((float)fb->buf[i] / 127.5f) - 1.0f;

            int32_t quantized_val = (int32_t)(normalized / scale + zero_point);
            
            if (quantized_val < -128) quantized_val = -128;
            if (quantized_val > 127) quantized_val = 127;

            input_buffer[i] = (int8_t)quantized_val;

            if (debug_pixel_waves < 3) {
                ESP_LOGI(TF_TAG, "Pixel [%d] -> Raw Cam Byte: %d | Normalized Float: %.4f | Final INT8: %d", 
                         debug_pixel_waves, fb->buf[i], normalized, input_buffer[i]);
                debug_pixel_waves++;
            }
        }
    } else {
        ESP_LOGE(TF_TAG, "Expected INT8 model execution layer.");
        return result;
    }

	if (global_interpreter->Invoke() != kTfLiteOk) return result;

    TfLiteTensor* output = global_interpreter->output(0);
    
    int8_t organic_raw = output->data.int8[0];
    int8_t recyclable_raw = output->data.int8[1];

    float organic_score = (organic_raw - output->params.zero_point) * output->params.scale;
    float recyclable_score = (recyclable_raw - output->params.zero_point) * output->params.scale;

    if (recyclable_score > organic_score) {
        result.identification = 1;
        result.confidence = (double)recyclable_score;
    } else {
        result.identification = 0;
        result.confidence = (double)organic_score;
    }

    return result;
}
