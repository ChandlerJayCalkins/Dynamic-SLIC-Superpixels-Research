cmake_minimum_required(VERSION 3.10)

project(demo)

set(OpenCV_DIR C:/Libraries/opencv_release/install)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/demo.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
