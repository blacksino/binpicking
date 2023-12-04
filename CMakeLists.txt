cmake_minimum_required(VERSION 3.20)
project(RGBDCamera)

set(CMAKE_CXX_STANDARD 11)
include_directories(dependency/include)
link_directories(dependency/lib)

find_package(OpenCV)
message(${OpenCV_LIBRARIES})

add_library(RGBDCameraSDK SHARED main.cpp)
target_link_libraries(RGBDCameraSDK Mv3dRgbd ${OpenCV_LIBRARIES})


