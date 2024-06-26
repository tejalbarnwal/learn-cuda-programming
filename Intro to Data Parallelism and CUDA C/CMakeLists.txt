cmake_minimum_required(VERSION 3.10)
project(intro_cuda)

## reference link: https://cmake.org/cmake/help/latest/policy/CMP0146.html
cmake_policy(SET CMP0146 NEW)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)
### use the below FindCUDA module for cmake version before 3.10
# find_package(CUDA REQUIRED)
# if (CUDA_FOUND)
#     message("CUDA found")
# else()
#     message("CUDA not found")
# endif()
# include_directories(${CUDA_INCLUDE_DIRS}) and target_link_libraries(${CUDA_LIBRARIES})
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})

add_executable(vectorAddCpp src/vectorAdd.cpp)

add_executable(vectorAddCuda src/vectorAddCUDA.cu)

add_executable(matricesAddCuda src/matricesAddCUDA.cu)
target_link_libraries(matricesAddCuda Eigen3::Eigen)

add_executable(matrices4AddCuda src/matrices4AddCUDA.cu)
target_link_libraries(matrices4AddCuda Eigen3::Eigen)
