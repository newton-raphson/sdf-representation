cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Set the Torch directory
set(Torch_DIR /work/mech-ai-scratch/samundra/packages/libtorch/share/cmake/Torch)


find_package(Torch REQUIRED)
add_executable(test_loading test_loading.cpp)
target_link_libraries(test_loading "${TORCH_LIBRARIES}")
set_property(TARGET test_loading PROPERTY CXX_STANDARD 17)

