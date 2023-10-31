cmake_minimum_required(VERSION 3.12)
project(OpenMPTest)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-fopenmp)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add include directories
include_directories(util)
include_directories(math)

# Add executable target
add_executable(OpenMPTest main.cpp)
add_subdirectory(../third_party/googletest ${CMAKE_BINARY_DIR}/googletest)

# Link OpenMP library
target_link_libraries(OpenMPTest PUBLIC OpenMP::OpenMP_CXX gtest gtest_main)
