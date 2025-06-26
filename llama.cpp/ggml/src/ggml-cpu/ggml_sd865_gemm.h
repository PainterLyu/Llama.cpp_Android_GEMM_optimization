#pragma once

#include "ggml.h"
#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// 前向声明
struct ggml_compute_params;

// 骁龙865专用GEMM优化算子
// 针对Snapdragon 865 (Cortex-A77 + A55) 架构优化
// 支持NEON和DOTPROD指令，但不支持i8mm

// 骁龙865硬件特性
#define SD865_L1_CACHE_SIZE (64 * 1024)     // 64KB L1 cache
#define SD865_L2_CACHE_SIZE (512 * 1024)    // 512KB L2 cache
#define SD865_L3_CACHE_SIZE (4 * 1024 * 1024) // 4MB L3 cache
#define SD865_CACHE_LINE_SIZE 64             // 64字节cache line
#define SD865_PREFETCH_DISTANCE 2            // 预取距离

// 优化参数
#define SD865_BLOCK_M 64                     // M维度分块大小
#define SD865_BLOCK_N 64                     // N维度分块大小  
#define SD865_BLOCK_K 256                    // K维度分块大小
#define SD865_MICRO_M 8                      // 微内核M维度
#define SD865_MICRO_N 8                      // 微内核N维度

// 全局控制变量（用于性能对比测试）
extern bool g_disable_sd865_optimization;

// 检测是否为骁龙865设备
bool ggml_sd865_is_supported(void);

// 骁龙865专用Q4_0×Q8_0点积函数
void ggml_vec_dot_q4_0_q8_0_sd865(
    int n,
    float * GGML_RESTRICT s,
    size_t bs,
    const void * GGML_RESTRICT vx,
    size_t bx,
    const void * GGML_RESTRICT vy,
    size_t by,
    int nrc
);

// 骁龙865专用Q8_0×Q8_0点积函数
void ggml_vec_dot_q8_0_q8_0_sd865(
    int n,
    float * GGML_RESTRICT s,
    size_t bs,
    const void * GGML_RESTRICT vx,
    size_t bx,
    const void * GGML_RESTRICT vy,
    size_t by,
    int nrc
);

// 骁龙865专用矩阵乘法微内核
void ggml_gemm_q4_0_q8_0_micro_kernel_sd865(
    int M, int N, int K,
    const void * GGML_RESTRICT A,
    const void * GGML_RESTRICT B,
    float * GGML_RESTRICT C,
    int lda, int ldb, int ldc
);

// 骁龙865专用Q8_0×Q8_0矩阵乘法微内核
void ggml_gemm_q8_0_q8_0_micro_kernel_sd865(
    int M, int N, int K,
    const void * GGML_RESTRICT A,
    const void * GGML_RESTRICT B,
    float * GGML_RESTRICT C,
    int lda, int ldb, int ldc
);

// 骁龙865专用分块矩阵乘法
void ggml_compute_forward_mul_mat_sd865(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

// 数据重排函数 - 优化内存访问模式
void ggml_sd865_repack_q4_0(
    const void * GGML_RESTRICT src,
    void * GGML_RESTRICT dst,
    int M, int K
);

void ggml_sd865_repack_q8_0(
    const void * GGML_RESTRICT src,
    void * GGML_RESTRICT dst,
    int K, int N
);

#ifdef __cplusplus
}
#endif
