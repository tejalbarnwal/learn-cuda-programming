cmake_minimum_required(VERSION 3.10)
project(intro_cuda)

## reference link: https://cmake.org/cmake/help/latest/policy/CMP0146.html
cmake_policy(SET CMP0146 NEW)
### use the below FindCUDA module for cmake version before 3.10
# find_package(CUDA REQUIRED)
# if (CUDA_FOUND)
#     message("CUDA found")
# else()
#     message("CUDA not found")
# endif()

add_executable(vectorAddCpp src/vectorAdd.cpp)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)
add_executable(vectorAddCuda src/vectorAddCUDA.cu)
