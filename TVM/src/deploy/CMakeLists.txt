cmake_minimum_required(VERSION 3.0)

project(TVM_GEMM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置TVM的安装路径
set(TVM_ROOT /home/wangbeichen/workspace/tvm)
file(GLOB TVM_3rd_ROOT ${TVM_ROOT}/3rdparty/*)

# 添加TVM库的头文件路径
include_directories(${TVM_ROOT}/include)

# 遍历所有文件夹
foreach(SUBDIR ${TVM_3rd_ROOT})
    if(IS_DIRECTORY ${SUBDIR})
        # 如果是文件夹，则添加到include_directories中
        include_directories(${SUBDIR}/include)
    endif()
endforeach()

# 添加TVM库的链接路径
link_directories(${TVM_ROOT}/build)

# 添加可执行文件
add_executable(gemm Gemm.cpp)

# 链接TVM库
target_link_libraries(gemm tvm_runtime)