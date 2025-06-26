#include "ggml_sd865_gemm.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <arm_neon.h>
#include <string.h>
#include <assert.h>

// 检测是否为骁龙865设备
bool ggml_sd865_is_supported(void) {
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

// 骁龙865专用Q8_0×Q8_0点积函数 - 深度优化版本
void ggml_vec_dot_q8_0_q8_0_sd865(
    int n,
    float * GGML_RESTRICT s,
    size_t bs,
    const void * GGML_RESTRICT vx,
    size_t bx,
    const void * GGML_RESTRICT vy,
    size_t by,
    int nrc) {

    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1); // 骁龙865不支持i8mm
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

    // 使用8路并行累加器提高ILP 
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);
    float32x4_t sumv2 = vdupq_n_f32(0.0f);
    float32x4_t sumv3 = vdupq_n_f32(0.0f);

    // 4路循环展开，提高吞吐量
    int i = 0;
    for (; i + 3 < nb; i += 4) {
        // 预取下4个数据块 
        if (i + SD865_PREFETCH_DISTANCE + 3 < nb) {
            __builtin_prefetch(&x[i + SD865_PREFETCH_DISTANCE], 0, 3);
            __builtin_prefetch(&y[i + SD865_PREFETCH_DISTANCE], 0, 3);
            __builtin_prefetch(&x[i + SD865_PREFETCH_DISTANCE + 2], 0, 3);
            __builtin_prefetch(&y[i + SD865_PREFETCH_DISTANCE + 2], 0, 3);
        }

       
        for (int j = 0; j < 4; j++) {
            const int idx = i + j;

            // 计算缩放因子
            const float d = x[idx].d * y[idx].d;
            const float32x4_t d_vec = vdupq_n_f32(d);

            // 加载Q8_0数据 
            const int8x16_t qx_0 = vld1q_s8(x[idx].qs);
            const int8x16_t qx_1 = vld1q_s8(x[idx].qs + 16);
            const int8x16_t qy_0 = vld1q_s8(y[idx].qs);
            const int8x16_t qy_1 = vld1q_s8(y[idx].qs + 16);

            // 使用DOTPROD指令进行4路并行计算
            const int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), qx_0, qy_0);
            const int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), qx_1, qy_1);

            // 累加到不同的累加器
            if (j == 0) {
                sumv0 = vfmaq_f32(sumv0, vcvtq_f32_s32(p0), d_vec);
                sumv1 = vfmaq_f32(sumv1, vcvtq_f32_s32(p1), d_vec);
            } else if (j == 1) {
                sumv2 = vfmaq_f32(sumv2, vcvtq_f32_s32(p0), d_vec);
                sumv3 = vfmaq_f32(sumv3, vcvtq_f32_s32(p1), d_vec);
            } else if (j == 2) {
                sumv0 = vfmaq_f32(sumv0, vcvtq_f32_s32(p0), d_vec);
                sumv1 = vfmaq_f32(sumv1, vcvtq_f32_s32(p1), d_vec);
            } else {
                sumv2 = vfmaq_f32(sumv2, vcvtq_f32_s32(p0), d_vec);
                sumv3 = vfmaq_f32(sumv3, vcvtq_f32_s32(p1), d_vec);
            }
        }
    }

    // 处理剩余的块
    for (; i < nb; ++i) {
        const float d = x[i].d * y[i].d;
        const float32x4_t d_vec = vdupq_n_f32(d);

        const int8x16_t qx_0 = vld1q_s8(x[i].qs);
        const int8x16_t qx_1 = vld1q_s8(x[i].qs + 16);
        const int8x16_t qy_0 = vld1q_s8(y[i].qs);
        const int8x16_t qy_1 = vld1q_s8(y[i].qs + 16);

        const int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), qx_0, qy_0);
        const int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), qx_1, qy_1);

        sumv0 = vfmaq_f32(sumv0, vcvtq_f32_s32(p0), d_vec);
        sumv1 = vfmaq_f32(sumv1, vcvtq_f32_s32(p1), d_vec);
    }

    // 最终求和 - 使用树形归约减少延迟
    const float32x4_t sum_pair0 = vaddq_f32(sumv0, sumv1);
    const float32x4_t sum_pair1 = vaddq_f32(sumv2, sumv3);
    const float32x4_t sum_all = vaddq_f32(sum_pair0, sum_pair1);

    *s = vaddvq_f32(sum_all);

#else
    // 回退到标准实现
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        int sumi = 0;
        for (int j = 0; j < qk; ++j) {
            sumi += x[i].qs[j] * y[i].qs[j];
        }
        sumf += sumi * x[i].d * y[i].d;
    }
    *s = sumf;
#endif
}

// Q8_0×Q8_0微内核 
void ggml_gemm_q8_0_q8_0_micro_kernel_sd865(
    int M, int N, int K,
    const void * GGML_RESTRICT A,
    const void * GGML_RESTRICT B,
    float * GGML_RESTRICT C,
    int lda, int ldb, int ldc) {

    assert(M <= SD865_MICRO_M);
    assert(N <= SD865_MICRO_N);

    const block_q8_0 * GGML_RESTRICT a_blocks = (const block_q8_0 *)A;
    const block_q8_0 * GGML_RESTRICT b_blocks = (const block_q8_0 *)B;
    
    
    float32x4_t acc[SD865_MICRO_M][SD865_MICRO_N/4];
    
  
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < (N+3)/4; j++) {
            acc[i][j] = vdupq_n_f32(0.0f);
        }
    }
    
    // 主计算循环
    const int nb = K / QK8_0;
    for (int k = 0; k < nb; k++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                const int a_idx = i * (K / QK8_0) + k;
                const int b_idx = j * (K / QK8_0) + k;
                
                float dot_result;
                ggml_vec_dot_q8_0_q8_0_sd865(
                    QK8_0, &dot_result, 0,
                    &a_blocks[a_idx], 0,
                    &b_blocks[b_idx], 0, 1
                );
                
                C[i * ldc + j] += dot_result;
            }
        }
    }
}

//三层分块
void ggml_compute_forward_mul_mat_sd865(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // 只处理Q8_0×Q8_0的情况
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_Q8_0) {
        return; 
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    // 矩阵维度
    const int M = ne01;  
    const int N = ne11;  
    const int K = ne00;  

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

// 数据重排
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
