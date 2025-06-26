#pragma once

#include "ggml.h"
#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif


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

// 检测
bool ggml_sd865_is_supported(void);

// Q8_0×Q8_0矩阵乘法微内核
void ggml_gemm_q8_0_q8_0_micro_kernel_sd865(
    int M, int N, int K,
    const void * GGML_RESTRICT A,
    const void * GGML_RESTRICT B,
    float * GGML_RESTRICT C,
    int lda, int ldb, int ldc
);

//分块矩阵乘法主函数
void ggml_compute_forward_mul_mat_sd865(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

// 数据重排
void ggml_sd865_repack_q8_0(
    const void * GGML_RESTRICT src,
    void * GGML_RESTRICT dst,
    int K, int N
);

#ifdef __cplusplus
}
#endif
