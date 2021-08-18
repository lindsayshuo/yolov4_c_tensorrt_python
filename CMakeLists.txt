cmake_minimum_required(VERSION 2.6)

project(yolov4)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)

# tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/home/lindsay/TensorRT-7.2.2.3/include)
link_directories(/home/lindsay/TensorRT-7.2.2.3/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu  ${PROJECT_SOURCE_DIR}/mish.cu)
target_link_libraries(myplugins nvinfer cudart)

# set(OpenCV_DIR /home/sz/opencv-master)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_library(yolov4_trt SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/mish.cu ${PROJECT_SOURCE_DIR}/yolov4_lib.cpp)
target_link_libraries(yolov4_trt nvinfer cudart ${OpenCV_LIBS})

add_executable(yolov4 ${PROJECT_SOURCE_DIR}/yolov4.cpp)

target_link_libraries(yolov4 nvinfer)
target_link_libraries(yolov4 cudart)
target_link_libraries(yolov4 myplugins)
target_link_libraries(yolov4 yolov4_trt)
target_link_libraries(yolov4 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

