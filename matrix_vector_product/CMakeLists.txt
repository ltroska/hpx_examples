#################################
#Compile matrix_vector_product
#@date 2016-05
#@author lukas.troska@gmail.com
#################################

#HPX requires this version
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

#C++-based project
project(matrix_vector_product CXX)

find_package(HPX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)

add_compile_options(-std=c++14)

# --------------- MAIN --------------- #
add_hpx_executable(
    matrix_vector_product
    ESSENTIAL
    SOURCES src/matrix_vector_product.cpp
)
