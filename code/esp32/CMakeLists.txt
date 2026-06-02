# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly

cmake_minimum_required(VERSION 4.1)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(AutoRecycle)

littlefs_create_partition_image(model "model" FLASH_IN_PROJECT)
