#include "ggml_sd865_gemm.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-cpu-impl.h"
#include "quants.h"

#include <arm_neon.h>
#include <string.h>
#include <assert.h>

// 全局变量控制优化启用/禁用（用于性能对比测试）
bool g_disable_sd865_optimization = false;

// 检测是否为骁龙865设备
bool ggml_sd865_is_supported(void) {
    if (g_disable_sd865_optimization) {
        return false;
    }

    static int cached_result = -1;
    if (cached_result == -1) {
        // 检查CPU特性：支持DOTPROD但不支持i8mm
        #if defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
        cached_result = 1;
        #else
        cached_result = 0;
        #endif
    }
    return cached_result == 1;
}

// 骁龙865专用分块矩阵乘法 - 三层分块策略
void ggml_compute_forward_mul_mat_sd865(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // 只处理Q8_0×Q8_0的情况，专注于cache优化
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_Q8_0) {
        return; // 让调用者回退到标准实现
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    // 矩阵维度
    const int M = ne01;  // src0的行数
    const int N = ne11;  // src1的列数
    const int K = ne00;  // 内积维度

    // 线程工作分配 - 按M维度分割
    const int M_per_thread = (M + nth - 1) / nth;
    const int M_start = ith * M_per_thread;
    const int M_end = MIN(M_start + M_per_thread, M);

    if (M_start >= M_end) return;

    // 三层分块策略：
    // L3: SD865_BLOCK_M x SD865_BLOCK_N x SD865_BLOCK_K (针对L3 cache)
    // L2: 微内核分块 (针对L2 cache)
    // L1: SIMD向量化 (针对L1 cache和寄存器)

    for (int m_block = M_start; m_block < M_end; m_block += SD865_BLOCK_M) {
        const int M_block_size = MIN(SD865_BLOCK_M, M_end - m_block);

        for (int n_block = 0; n_block < N; n_block += SD865_BLOCK_N) {
            const int N_block_size = MIN(SD865_BLOCK_N, N - n_block);

            // 初始化C块为0
            for (int i = 0; i < M_block_size; i++) {
                for (int j = 0; j < N_block_size; j++) {
                    const int dst_idx = (m_block + i) * ne11 + (n_block + j);
                    ((float*)dst->data)[dst_idx] = 0.0f;
                }
            }

            for (int k_block = 0; k_block < K; k_block += SD865_BLOCK_K) {
                const int K_block_size = MIN(SD865_BLOCK_K, K - k_block);

                // 微内核分块
                for (int m_micro = 0; m_micro < M_block_size; m_micro += SD865_MICRO_M) {
                    const int M_micro_size = MIN(SD865_MICRO_M, M_block_size - m_micro);

                    for (int n_micro = 0; n_micro < N_block_size; n_micro += SD865_MICRO_N) {
                        const int N_micro_size = MIN(SD865_MICRO_N, N_block_size - n_micro);

                        // 计算数据指针
                        const int m_global = m_block + m_micro;
                        const int n_global = n_block + n_micro;
                        const int k_global = k_block;

                        // A矩阵指针 (Q8_0格式)
                        const void * A_ptr = (const char*)src0->data +
                            (m_global * nb01 + (k_global / QK8_0) * sizeof(block_q8_0));

                        // B矩阵指针 (Q8_0格式)
                        const void * B_ptr = (const char*)src1->data +
                            (n_global * nb11 + (k_global / QK8_0) * sizeof(block_q8_0));

                        // C矩阵指针
                        float * C_ptr = (float*)dst->data + m_global * ne11 + n_global;

                        // 调用Q8_0×Q8_0微内核
                        ggml_gemm_q8_0_q8_0_micro_kernel_sd865(
                            M_micro_size, N_micro_size, K_block_size,
                            A_ptr, B_ptr, C_ptr,
                            K / QK8_0, K / QK8_0, ne11
                        );
                    }
                }
            }
        }
    }
}


void ggml_sd865_repack_q8_0(
    const void * GGML_RESTRICT src,
    void * GGML_RESTRICT dst,
    int K, int N) {

    const block_q8_0 * GGML_RESTRICT src_blocks = (const block_q8_0 *)src;
    block_q8_0 * GGML_RESTRICT dst_blocks = (block_q8_0 *)dst;

    const int K_blocks = K / QK8_0;

    // 按列主序重排为行主序，提高cache效率
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K_blocks; k++) {
            const int src_idx = n * K_blocks + k;
            const int dst_idx = (n / SD865_MICRO_N) * (SD865_MICRO_N * K_blocks) +
                               (k / 4) * (SD865_MICRO_N * 4) +
                               (n % SD865_MICRO_N) * 4 + (k % 4);
            dst_blocks[dst_idx] = src_blocks[src_idx];
        }
    }
}

// 骁龙865专用Q8_0×Q8_0微内核 - 三级cache优化
void ggml_gemm_q8_0_q8_0_micro_kernel_sd865(
    int M, int N, int K,
    const void * GGML_RESTRICT A,
    const void * GGML_RESTRICT B,
    float * GGML_RESTRICT C,
    int lda, int ldb, int ldc) {

    const block_q8_0 * GGML_RESTRICT a_blocks = (const block_q8_0*)A;
    const block_q8_0 * GGML_RESTRICT b_blocks = (const block_q8_0*)B;

    // 高效向量化实现 - 参考原生ggml的优化策略
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

    const int nb = K / QK8_0;  // block数量

    // 使用临时缓冲区，参考原生实现的策略
    float tmp[SD865_MICRO_M * SD865_MICRO_N];

    // 按原生实现的模式：先计算到临时缓冲区，再批量写回
    for (int j = 0; j < N; j++) {
        // 预取B矩阵数据
        if (j + 1 < N) {
            for (int k = 0; k < MIN(4, nb); k++) {
                __builtin_prefetch(&b_blocks[(j + 1) * ldb + k], 0, 3);
            }
        }

        // 处理当前列的所有行
        for (int i = 0; i < M; i++) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);

            // K维度向量化处理，4个block一组
            int k;
            for (k = 0; k + 3 < nb; k += 4) {
                // 加载A矩阵的4个连续blocks
                const block_q8_0 * a_ptr = &a_blocks[i * lda + k];
                const block_q8_0 * b_ptr = &b_blocks[j * ldb + k];

                // 4路并行DOTPROD计算
                float results[4];
                for (int kk = 0; kk < 4; kk++) {
                    const float a_scale = GGML_FP16_TO_FP32(a_ptr[kk].d);
                    const float b_scale = GGML_FP16_TO_FP32(b_ptr[kk].d);

                    // 加载量化数据
                    const int8x16_t a_0 = vld1q_s8(a_ptr[kk].qs);
                    const int8x16_t a_1 = vld1q_s8(a_ptr[kk].qs + 16);
                    const int8x16_t b_0 = vld1q_s8(b_ptr[kk].qs);
                    const int8x16_t b_1 = vld1q_s8(b_ptr[kk].qs + 16);

                    // DOTPROD计算
                    const int32x4_t dot_0 = vdotq_s32(vdupq_n_s32(0), a_0, b_0);
                    const int32x4_t dot_1 = vdotq_s32(vdupq_n_s32(0), a_1, b_1);
                    const int32x4_t dot_sum = vaddq_s32(dot_0, dot_1);

                    // 应用缩放并累加
                    results[kk] = vaddvq_s32(dot_sum) * a_scale * b_scale;
                }

                // 向量化累加
                const float32x4_t results_vec = vld1q_f32(results);
                sum_vec = vaddq_f32(sum_vec, results_vec);
            }

            // 处理剩余的blocks
            float scalar_sum = vaddvq_f32(sum_vec);
            for (; k < nb; k++) {
                const block_q8_0 * a_block = &a_blocks[i * lda + k];
                const block_q8_0 * b_block = &b_blocks[j * ldb + k];

                const float a_scale = GGML_FP16_TO_FP32(a_block->d);
                const float b_scale = GGML_FP16_TO_FP32(b_block->d);

                const int8x16_t a_0 = vld1q_s8(a_block->qs);
                const int8x16_t a_1 = vld1q_s8(a_block->qs + 16);
                const int8x16_t b_0 = vld1q_s8(b_block->qs);
                const int8x16_t b_1 = vld1q_s8(b_block->qs + 16);

                const int32x4_t dot_0 = vdotq_s32(vdupq_n_s32(0), a_0, b_0);
                const int32x4_t dot_1 = vdotq_s32(vdupq_n_s32(0), a_1, b_1);
                const int32x4_t dot_sum = vaddq_s32(dot_0, dot_1);

                scalar_sum += vaddvq_s32(dot_sum) * a_scale * b_scale;
            }

            // 存储到临时缓冲区
            tmp[i * N + j] = scalar_sum;
        }
    }

    // 批量写回到C矩阵，参考原生实现的memcpy策略
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += tmp[i * N + j];
        }
    }

#else
    // 回退到标量实现
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k += QK8_0) {
                const int a_idx = i * lda + k / QK8_0;
                const int b_idx = j * ldb + k / QK8_0;

                float dot_result;
                ggml_vec_dot_q8_0_q8_0(
                    QK8_0, &dot_result, 0,
                    &a_blocks[a_idx], 0,
                    &b_blocks[b_idx], 0, 1
                );
                sum += dot_result;
            }
            C[i * ldc + j] += sum;
        }
    }
#endif
}
