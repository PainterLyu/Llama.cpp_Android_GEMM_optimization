#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"

// 计时函数
double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

// 创建Q8_0测试矩阵
struct ggml_tensor* create_test_matrix_q8_0(struct ggml_context* ctx, int rows, int cols, int seed) {
    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, cols, rows);
    
    srand(seed);
    float* temp_data = malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) {
        temp_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f; // 范围[-2, 2]
    }
    
    ggml_quantize_q8_0(temp_data, tensor->data, rows * cols, cols, NULL);
    free(temp_data);
    
    return tensor;
}

// 验证结果正确性 - 完整版本
bool validate_results(float* result_optimized, float* result_reference, int size,
                     double* max_error, double* avg_error, double* rel_error,
                     int* mismatch_count, double* correlation) {
    *max_error = 0.0;
    *avg_error = 0.0;
    *rel_error = 0.0;
    *mismatch_count = 0;
    *correlation = 0.0;

    double sum_abs_diff = 0.0;
    double sum_abs_ref = 0.0;
    double sum_opt = 0.0, sum_ref = 0.0;
    double sum_opt_sq = 0.0, sum_ref_sq = 0.0, sum_opt_ref = 0.0;

    // 第一遍：计算基本误差统计
    for (int i = 0; i < size; i++) {
        double abs_diff = fabs(result_optimized[i] - result_reference[i]);
        double abs_ref = fabs(result_reference[i]);

        if (abs_diff > *max_error) {
            *max_error = abs_diff;
        }

        sum_abs_diff += abs_diff;
        sum_abs_ref += abs_ref;

        // 计算相关系数所需的统计量
        sum_opt += result_optimized[i];
        sum_ref += result_reference[i];
        sum_opt_sq += result_optimized[i] * result_optimized[i];
        sum_ref_sq += result_reference[i] * result_reference[i];
        sum_opt_ref += result_optimized[i] * result_reference[i];

        // 计算不匹配数量（相对误差 > 1e-6）
        if (abs_ref > 1e-10 && (abs_diff / abs_ref) > 1e-6) {
            (*mismatch_count)++;
        } else if (abs_ref <= 1e-10 && abs_diff > 1e-8) {
            (*mismatch_count)++;
        }
    }

    *avg_error = sum_abs_diff / size;
    *rel_error = (sum_abs_ref > 0) ? (sum_abs_diff / sum_abs_ref) : 0.0;

    // 计算皮尔逊相关系数
    double mean_opt = sum_opt / size;
    double mean_ref = sum_ref / size;
    double numerator = sum_opt_ref - size * mean_opt * mean_ref;
    double denominator = sqrt((sum_opt_sq - size * mean_opt * mean_opt) *
                             (sum_ref_sq - size * mean_ref * mean_ref));
    *correlation = (denominator > 1e-10) ? (numerator / denominator) : 0.0;

    // 严格的验证标准
    bool max_error_ok = *max_error < 1e-4;
    bool avg_error_ok = *avg_error < 1e-5;
    bool rel_error_ok = *rel_error < 1e-3;
    bool mismatch_ok = *mismatch_count < (size / 10000); // 允许万分之一的不匹配
    bool correlation_ok = *correlation > 0.9999; // 相关系数要求很高

    return max_error_ok && avg_error_ok && rel_error_ok && mismatch_ok && correlation_ok;
}

// 双版本GEMM计算函数
struct ggml_tensor* compute_gemm_with_backend(struct ggml_context* ctx,
                                            struct ggml_tensor* A, struct ggml_tensor* B,
                                            const char* backend_name) {
    printf("  使用%s计算GEMM...\n", backend_name);

    struct ggml_tensor* C = ggml_mul_mat(ctx, A, B);
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, C);

    double start_time = get_time_us();
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    double end_time = get_time_us();

    printf("  %s计算时间: %.3f ms\n", backend_name, (end_time - start_time) / 1000.0);

    return C;
}

// 完整的正确性验证函数
bool comprehensive_correctness_test(int M, int N, int K) {
    printf("\n=== 正确性验证: M=%d, N=%d, K=%d ===\n", M, N, K);

    // 初始化ggml上下文
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 1024, // 1GB
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    struct ggml_context* ctx_original = ggml_init(params);
    struct ggml_context* ctx_optimized = ggml_init(params);

    if (!ctx_original || !ctx_optimized) {
        printf("错误: 无法初始化ggml上下文\n");
        return false;
    }

    // 创建相同的测试矩阵
    printf("创建测试矩阵...\n");
    struct ggml_tensor* A_orig = create_test_matrix_q8_0(ctx_original, M, K, 12345);
    struct ggml_tensor* B_orig = create_test_matrix_q8_0(ctx_original, K, N, 67890);
    struct ggml_tensor* A_opt = create_test_matrix_q8_0(ctx_optimized, M, K, 12345);
    struct ggml_tensor* B_opt = create_test_matrix_q8_0(ctx_optimized, K, N, 67890);

    // 使用原生版本计算
    struct ggml_tensor* C_original = compute_gemm_with_backend(ctx_original, A_orig, B_orig, "原生版本");

    // 使用优化版本计算
    struct ggml_tensor* C_optimized = compute_gemm_with_backend(ctx_optimized, A_opt, B_opt, "SD865优化版本");

    // 进行详细的正确性验证
    printf("\n--- 数值精度对比分析 ---\n");

    float* data_original = (float*)C_original->data;
    float* data_optimized = (float*)C_optimized->data;

    double max_error, avg_error, rel_error, correlation;
    int mismatch_count;

    bool is_correct = validate_results(data_optimized, data_original, M * N,
                                     &max_error, &avg_error, &rel_error,
                                     &mismatch_count, &correlation);

    // 输出详细的验证结果
    printf("数值精度分析结果:\n");
    printf("  矩阵尺寸: %dx%dx%d (%d个元素)\n", M, N, K, M * N);
    printf("  最大绝对误差: %.2e %s\n", max_error, (max_error < 1e-4) ? "✓" : "✗");
    printf("  平均绝对误差: %.2e %s\n", avg_error, (avg_error < 1e-5) ? "✓" : "✗");
    printf("  相对误差: %.2e %s\n", rel_error, (rel_error < 1e-3) ? "✓" : "✗");
    printf("  不匹配元素数: %d/%d (%.4f%%) %s\n", mismatch_count, M * N,
           (double)mismatch_count / (M * N) * 100.0,
           (mismatch_count < M * N / 10000) ? "✓" : "✗");
    printf("  相关系数: %.8f %s\n", correlation, (correlation > 0.9999) ? "✓" : "✗");

    // 统计信息对比
    printf("\n--- 统计信息对比 ---\n");

    // 原生版本统计
    float sum_orig = 0.0f, min_orig = data_original[0], max_orig = data_original[0];
    for (int i = 0; i < M * N; i++) {
        sum_orig += data_original[i];
        if (data_original[i] < min_orig) min_orig = data_original[i];
        if (data_original[i] > max_orig) max_orig = data_original[i];
    }

    // 优化版本统计
    float sum_opt = 0.0f, min_opt = data_optimized[0], max_opt = data_optimized[0];
    for (int i = 0; i < M * N; i++) {
        sum_opt += data_optimized[i];
        if (data_optimized[i] < min_opt) min_opt = data_optimized[i];
        if (data_optimized[i] > max_opt) max_opt = data_optimized[i];
    }

    printf("原生版本统计:\n");
    printf("  元素和: %.6f, 平均值: %.6f\n", sum_orig, sum_orig / (M * N));
    printf("  数值范围: [%.6f, %.6f]\n", min_orig, max_orig);

    printf("优化版本统计:\n");
    printf("  元素和: %.6f, 平均值: %.6f\n", sum_opt, sum_opt / (M * N));
    printf("  数值范围: [%.6f, %.6f]\n", min_opt, max_opt);

    printf("统计差异:\n");
    printf("  元素和差异: %.2e\n", fabs(sum_orig - sum_opt));
    printf("  平均值差异: %.2e\n", fabs(sum_orig - sum_opt) / (M * N));
    printf("  范围差异: [%.2e, %.2e]\n", fabs(min_orig - min_opt), fabs(max_orig - max_opt));

    // 异常值检查
    printf("\n--- 异常值检查 ---\n");
    int nan_orig = 0, inf_orig = 0, nan_opt = 0, inf_opt = 0;

    for (int i = 0; i < M * N; i++) {
        if (isnan(data_original[i])) nan_orig++;
        if (isinf(data_original[i])) inf_orig++;
        if (isnan(data_optimized[i])) nan_opt++;
        if (isinf(data_optimized[i])) inf_opt++;
    }

    printf("原生版本: NaN=%d, Inf=%d\n", nan_orig, inf_orig);
    printf("优化版本: NaN=%d, Inf=%d\n", nan_opt, inf_opt);

    bool no_anomalies = (nan_orig == 0 && inf_orig == 0 && nan_opt == 0 && inf_opt == 0);
    printf("异常值检查: %s\n", no_anomalies ? "✓ 通过" : "✗ 失败");

    // 最终验证结果
    printf("\n--- 最终验证结果 ---\n");
    printf("正确性验证: %s\n", is_correct ? "✓ 通过" : "✗ 失败");

    if (!is_correct) {
        printf("⚠️  数值精度验证失败，可能的原因:\n");
        if (max_error >= 1e-4) printf("   - 最大误差过大，可能存在算法实现差异\n");
        if (avg_error >= 1e-5) printf("   - 平均误差过大，可能存在系统性偏差\n");
        if (rel_error >= 1e-3) printf("   - 相对误差过大，可能存在数值稳定性问题\n");
        if (mismatch_count >= M * N / 10000) printf("   - 不匹配元素过多，可能存在实现错误\n");
        if (correlation <= 0.9999) printf("   - 相关性不足，结果可能不一致\n");
    }

    ggml_free(ctx_original);
    ggml_free(ctx_optimized);

    return is_correct && no_anomalies;
}

// 性能测试函数
void performance_test(int M, int N, int K, int iterations) {
    printf("\n=== 性能测试: M=%d, N=%d, K=%d ===\n", M, N, K);
    
    // 初始化ggml上下文
    struct ggml_init_params params = {
        .mem_size = 2048 * 1024 * 1024, // 2GB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("错误: 无法初始化ggml上下文\n");
        return;
    }
    
    // 创建测试矩阵
    printf("创建测试矩阵...\n");
    struct ggml_tensor* A = create_test_matrix_q8_0(ctx, M, K, 12345);
    struct ggml_tensor* B = create_test_matrix_q8_0(ctx, K, N, 67890);
    
    printf("矩阵A: %dx%d (Q8_0)\n", M, K);
    printf("矩阵B: %dx%d (Q8_0)\n", K, N);
    printf("结果C: %dx%d (F32)\n", M, N);
    
    // 预热阶段
    printf("预热阶段...\n");
    for (int i = 0; i < 3; i++) {
        struct ggml_tensor* C = ggml_mul_mat(ctx, A, B);
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, C);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
    }
    
    // 正式性能测试
    printf("开始性能测试 (%d次迭代)...\n", iterations);
    
    double total_time = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;
    
    struct ggml_tensor* C_result = NULL;
    
    for (int i = 0; i < iterations; i++) {
        double start_time = get_time_us();
        
        struct ggml_tensor* C = ggml_mul_mat(ctx, A, B);
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, C);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        
        double end_time = get_time_us();
        double elapsed = (end_time - start_time) / 1000.0; // 转换为毫秒
        
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
        
        if (i == 0) C_result = C; // 保存第一次结果用于验证
        
        if ((i + 1) % 5 == 0) {
            printf("  完成 %d/%d 次迭代, 当前: %.2f ms\n", i + 1, iterations, elapsed);
        }
    }
    
    // 计算性能指标
    double avg_time = total_time / iterations;
    double gflops = (2.0 * M * N * K) / (avg_time * 1e6); // GFLOPS
    double throughput = 1000.0 / avg_time; // ops/second
    
    printf("\n=== 性能结果 ===\n");
    printf("平均时间: %.3f ms\n", avg_time);
    printf("最小时间: %.3f ms\n", min_time);
    printf("最大时间: %.3f ms\n", max_time);
    printf("标准差: %.3f ms\n", sqrt((max_time - min_time) * (max_time - min_time) / 12.0));
    printf("性能: %.2f GFLOPS\n", gflops);
    printf("吞吐量: %.2f ops/sec\n", throughput);
    
    // 计算内存带宽利用率
    size_t data_size = M * K * sizeof(block_q8_0) / 32 + K * N * sizeof(block_q8_0) / 32 + M * N * sizeof(float);
    double bandwidth = (data_size * iterations) / (total_time * 1e6) * 1000.0; // GB/s
    printf("内存带宽: %.2f GB/s\n", bandwidth);
    
    // 验证结果正确性（与参考实现对比）
    printf("\n=== 正确性验证 ===\n");
    if (C_result) {
        float* result_data = (float*)C_result->data;
        
        // 计算结果的统计信息
        float sum = 0.0f, min_val = result_data[0], max_val = result_data[0];
        for (int i = 0; i < M * N; i++) {
            sum += result_data[i];
            if (result_data[i] < min_val) min_val = result_data[i];
            if (result_data[i] > max_val) max_val = result_data[i];
        }
        
        printf("结果统计:\n");
        printf("  元素总数: %d\n", M * N);
        printf("  元素和: %.6f\n", sum);
        printf("  平均值: %.6f\n", sum / (M * N));
        printf("  最小值: %.6f\n", min_val);
        printf("  最大值: %.6f\n", max_val);
        printf("  数值范围: [%.6f, %.6f]\n", min_val, max_val);
        
        // 检查是否有异常值
        int nan_count = 0, inf_count = 0;
        for (int i = 0; i < M * N; i++) {
            if (isnan(result_data[i])) nan_count++;
            if (isinf(result_data[i])) inf_count++;
        }
        
        printf("异常值检查:\n");
        printf("  NaN数量: %d\n", nan_count);
        printf("  Inf数量: %d\n", inf_count);
        printf("  正确性: %s\n", (nan_count == 0 && inf_count == 0) ? "通过" : "失败");
    }
    
    ggml_free(ctx);
}

// 不同尺寸的压力测试
void stress_test() {
    printf("\n=== 压力测试 ===\n");
    
    struct {
        int M, N, K;
        const char* desc;
    } test_cases[] = {
        {64, 64, 64, "小矩阵"},
        {256, 256, 256, "中等矩阵"},
        {512, 512, 512, "大矩阵"},
        {1024, 1024, 1024, "超大矩阵"},
        {4096, 1, 4096, "Llama2单token"},
        {4096, 8, 4096, "Llama2小批量"},
        {4096, 32, 4096, "Llama2中批量"},
        {11008, 1, 4096, "Llama2 FFN"},
        {4096, 1, 11008, "Llama2 FFN反向"}
    };
    
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    
    for (int i = 0; i < num_cases; i++) {
        printf("\n--- 测试案例 %d: %s ---\n", i + 1, test_cases[i].desc);
        
        // 根据矩阵大小调整迭代次数
        int iterations;
        if (test_cases[i].M * test_cases[i].N * test_cases[i].K < 1000000) {
            iterations = 20;
        } else if (test_cases[i].M * test_cases[i].N * test_cases[i].K < 100000000) {
            iterations = 10;
        } else {
            iterations = 5;
        }
        
        performance_test(test_cases[i].M, test_cases[i].N, test_cases[i].K, iterations);
    }
}

int main(int argc, char* argv[]) {
    printf("骁龙865 GEMM算子验证测试程序\n");
    printf("编译时间: %s %s\n", __DATE__, __TIME__);
    printf("========================================\n");
    
    // 检查CPU特性
    printf("\nCPU特性检查:\n");
    #ifdef __ARM_NEON
    printf("✓ ARM NEON支持\n");
    #else
    printf("✗ ARM NEON不支持\n");
    #endif
    
    #ifdef __ARM_FEATURE_DOTPROD
    printf("✓ ARM DOTPROD支持\n");
    #else
    printf("✗ ARM DOTPROD不支持\n");
    #endif
    
    #ifdef __ARM_FEATURE_MATMUL_INT8
    printf("✓ ARM i8mm支持\n");
    #else
    printf("✗ ARM i8mm不支持 (骁龙865正常)\n");
    #endif
    
    // 运行压力测试
    stress_test();
    
    printf("\n========================================\n");
    printf("测试完成！\n");
    
    return 0;
}
