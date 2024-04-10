cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(GEMM VERSION 1.0)

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CUDA_COMPILER  /usr/local/cuda-11.1/bin/nvcc)

set(CUDA_ARCHITECTURES 86)
# Check for CUDA
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-extended-lambda;-G)

# Enable CUDA for the project
enable_language(CUDA)

# Set source files
set(SOURCE_FILES
    main.cpp 
    GEMM.cu
    # Add other .cpp and .cu files here
)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cublas)

set_target_properties(${PROJECT_NAME} PROPERTIES
    # CUDA_SEPARABLE_COMPILATION ON # 允许分离编译
    CUDA_ARCHITECTURES 86
)
find_package(OpenMP)

# 如果 OpenMP 可用，设置编译选项
if(OPENMP_FOUND)
    # 设置 OpenMP 编译选项
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_C)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()