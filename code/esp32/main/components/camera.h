#ifndef CODE_CAMERA_H
#define CODE_CAMERA_H

#include <stdbool.h>
#include "esp_camera.h"

// Pin Definitions matching your schematic precisely
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define PCLK_GPIO_NUM     13
#define VSYNC_GPIO_NUM     6
#define HREF_GPIO_NUM      7

#define SIOD_GPIO_NUM      4  // CAM_SIOD
#define SIOC_GPIO_NUM      5  // CAM_SIOC

#define Y2_GPIO_NUM       11  // CAM_Y2
#define Y3_GPIO_NUM        9  // CAM_Y3
#define Y4_GPIO_NUM        8  // CAM_Y4
#define Y5_GPIO_NUM       10  // CAM_Y5
#define Y6_GPIO_NUM       12  // CAM_Y6
#define Y7_GPIO_NUM       18  // CAM_Y7
#define Y8_GPIO_NUM       17  // CAM_Y8
#define Y9_GPIO_NUM       16  // CAM_Y9

// Function declarations
bool initalize_camera(void);
camera_fb_t* get_frame(void);

#endif // CODE_CAMERA_H
