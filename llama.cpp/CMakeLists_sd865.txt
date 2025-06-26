cmake_minimum_required(VERSION 3.14)
project(sd865_gemm_test)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Android NDK配置
if(ANDROID)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod")
    
    # 启用DOTPROD支持
    add_definitions(-DGGML_USE_DOTPROD)
    add_definitions(-D__ARM_FEATURE_DOTPROD=1)
    
    # 优化标志 - 移除ffast-math避免编译错误
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fno-finite-math-only")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-finite-math-only")
    
    # 禁用不需要的功能
    set(GGML_OPENMP OFF)
    set(GGML_LLAMAFILE OFF)
    set(LLAMA_CURL OFF)
endif()

# 包含ggml源码
set(GGML_SOURCES
    ggml/src/ggml.c
    ggml/src/ggml-alloc.c
    ggml/src/ggml-backend.c
    ggml/src/ggml-quants.c
    ggml/src/ggml-cpu/ggml-cpu.c
    ggml/src/ggml-cpu/arch/arm/quants.c
    ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp
    ggml/src/ggml-cpu/vec.cpp
    ggml/src/ggml-cpu/traits.cpp
    ggml_sd865_gemm.c
)

# 包含目录
include_directories(
    ggml/include
    ggml/src
    ggml/src/ggml-cpu
)

# 创建ggml库
add_library(ggml_sd865 STATIC ${GGML_SOURCES})

# 设置编译选项
target_compile_definitions(ggml_sd865 PRIVATE
    GGML_USE_CPU
    GGML_USE_DOTPROD
)

if(ANDROID)
    target_compile_definitions(ggml_sd865 PRIVATE
        __ARM_FEATURE_DOTPROD=1
    )
endif()

# 创建测试程序
add_executable(test_sd865_gemm test_sd865_gemm.c)
target_link_libraries(test_sd865_gemm ggml_sd865)

# 创建基准测试程序
add_executable(benchmark_gemm benchmark_gemm.c)
target_link_libraries(benchmark_gemm ggml_sd865)

if(ANDROID)
    target_link_libraries(test_sd865_gemm log)
endif()

# 编译标志
target_compile_options(ggml_sd865 PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wcast-qual
    -Wno-unused-function
)

target_compile_options(test_sd865_gemm PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
