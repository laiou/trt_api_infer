cmake_minimum_required(VERSION 2.6)

project(project)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


cuda_add_library(yoloplugin SHARED ${PROJECT_SOURCE_DIR}/plugin/yoloplugin.cu)
target_link_libraries(yoloplugin nvinfer cudart ${OpenCV_LIBS})

add_executable(project ${PROJECT_SOURCE_DIR}/calib/calibrator.cpp ${PROJECT_SOURCE_DIR}/infer/infer.cpp)
target_link_libraries(project nvinfer)
target_link_libraries(project cudart)
target_link_libraries(project yoloplugin)
target_link_libraries(project ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

