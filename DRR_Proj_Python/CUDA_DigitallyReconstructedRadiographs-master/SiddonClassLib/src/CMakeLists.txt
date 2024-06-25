cmake_minimum_required(VERSION 3.16)
project(SiddonGpu)

SET(CMAKE_BUILD_TYPE "Release")
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    message(STATUS "optional:-fPIC")
endif(CMAKE_COMPILER_IS_GNUCXX)


find_package(CUDA REQUIRED)

cuda_add_library(SiddonGpu SiddonLib/siddon_class.cu SiddonLib/siddon_class.cuh)
target_link_libraries(SiddonGpu cudart cudadevrt)