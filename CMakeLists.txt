cmake_minimum_required(VERSION 2.8.12)
project(ImRegionalMax)

find_package(OpenCV REQUIRED core imgproc highgui) 

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ImRegionalMax ImRegionalMax.cpp)
target_link_libraries(ImRegionalMax ${OpenCV_LIBS})

