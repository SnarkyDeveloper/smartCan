// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdint.h>

typedef struct {
    double confidence;
    int identification; // 1 = Recyclable, 0 = Organic
} Identification;

#ifdef __cplusplus
extern "C" {
#endif

void* load_model_from_file(FILE* file_ptr, uint8_t** model_buffer_out);

Identification run_inference(void* model_ptr, void* camera_fb_ptr);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H
