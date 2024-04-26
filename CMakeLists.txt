cmake_minimum_required(VERSION 3.10)

project(tensor-cpp VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(tensor-cpp INTERFACE)
target_include_directories(tensor-cpp 
                            INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/tensor-cpp>)

add_subdirectory(apps)