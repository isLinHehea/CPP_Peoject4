# project
cmake_minimum_required(VERSION 3.12)

add_definitions(-DWITH_AVX2 -DWITH_OPENBLAS)

set(CMAKE_C_STANDARD 11)

project(Matrix)

add_compile_options(-mfma -lopenblas -fopenmp -mavx2)

add_compile_options(-O3)

ADD_EXECUTABLE(matrix matrix.c matrixmultiplication.c matrix.h)

target_link_libraries(matrix PUBLIC openblas)

find_package(OpenMP)
if(OpenMP_C_FOUND)
    message("OpenMP found.")
    target_link_libraries(matrix PUBLIC OpenMP::OpenMP_C)
endif()