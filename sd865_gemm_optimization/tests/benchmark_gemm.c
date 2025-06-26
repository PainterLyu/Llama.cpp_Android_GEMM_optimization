#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml_sd865_gemm.h"

// 计时函数
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// 创建Q8_0测试矩阵
struct ggml_tensor* create_test_matrix_q8_0(struct ggml_context* ctx, int rows, int cols) {
    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, cols, rows);
    
    // 填充随机数据
    float* temp_data = malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) {
        temp_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    
    // 量化为Q8_0
    ggml_quantize_q8_0(temp_data, tensor->data, rows * cols, cols, NULL);
    free(temp_data);
    
    return tensor;
}

// 禁用SD865优化的标志
extern bool g_disable_sd865_optimization;

// 性能对比测试
void benchmark_q8_0_gemm(int M, int N, int K, int iterations) {
    printf("=== Q8_0×Q8_0 GEMM性能对比测试 ===\n");
    printf("矩阵尺寸: M=%d, N=%d, K=%d\n", M, N, K);
    printf("迭代次数: %d\n\n", iterations);
    
    // 初始化ggml
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 1024, // 1GB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to initialize ggml context\n");
        return;
    }
    
    // 创建测试矩阵
    printf("创建Q8_0测试矩阵...\n");
    struct ggml_tensor* a = create_test_matrix_q8_0(ctx, M, K);  // Q8_0权重矩阵
    struct ggml_tensor* b = create_test_matrix_q8_0(ctx, K, N);  // Q8_0输入矩阵
    
    printf("检查SD865优化支持: %s\n", ggml_sd865_is_supported() ? "是" : "否");
    
    // 测试1：原始实现（禁用SD865优化）
    printf("\n=== 测试原始实现 ===\n");
    g_disable_sd865_optimization = true;
    
    // 预热
    for (int i = 0; i < 3; i++) {
        struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
    }
    
    double original_total_time = 0.0;
    double original_min_time = 1e9;
    double original_max_time = 0.0;
    
    for (int i = 0; i < iterations; i++) {
        double start_time = get_time_ms();
        
        struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        
        double end_time = get_time_ms();
        double elapsed = end_time - start_time;
        
        original_total_time += elapsed;
        if (elapsed < original_min_time) original_min_time = elapsed;
        if (elapsed > original_max_time) original_max_time = elapsed;
        
        if ((i + 1) % 5 == 0) {
            printf("原始实现: 完成 %d/%d 次迭代\n", i + 1, iterations);
        }
    }
    
    double original_avg_time = original_total_time / iterations;
    double original_gflops = (2.0 * M * N * K) / (original_avg_time * 1e6);
    
    printf("原始实现结果:\n");
    printf("  平均时间: %.2f ms\n", original_avg_time);
    printf("  最小时间: %.2f ms\n", original_min_time);
    printf("  最大时间: %.2f ms\n", original_max_time);
    printf("  性能: %.2f GFLOPS\n", original_gflops);
    
    // 测试2：SD865优化实现
    printf("\n=== 测试SD865优化实现 ===\n");
    g_disable_sd865_optimization = false;
    
    // 预热
    for (int i = 0; i < 3; i++) {
        struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
    }
    
    double optimized_total_time = 0.0;
    double optimized_min_time = 1e9;
    double optimized_max_time = 0.0;
    
    for (int i = 0; i < iterations; i++) {
        double start_time = get_time_ms();
        
        struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        
        double end_time = get_time_ms();
        double elapsed = end_time - start_time;
        
        optimized_total_time += elapsed;
        if (elapsed < optimized_min_time) optimized_min_time = elapsed;
        if (elapsed > optimized_max_time) optimized_max_time = elapsed;
        
        if ((i + 1) % 5 == 0) {
            printf("优化实现: 完成 %d/%d 次迭代\n", i + 1, iterations);
        }
    }
    
    double optimized_avg_time = optimized_total_time / iterations;
    double optimized_gflops = (2.0 * M * N * K) / (optimized_avg_time * 1e6);
    
    printf("优化实现结果:\n");
    printf("  平均时间: %.2f ms\n", optimized_avg_time);
    printf("  最小时间: %.2f ms\n", optimized_min_time);
    printf("  最大时间: %.2f ms\n", optimized_max_time);
    printf("  性能: %.2f GFLOPS\n", optimized_gflops);
    
    // 计算性能提升
    double speedup = original_avg_time / optimized_avg_time;
    printf("\n=== 性能对比结果 ===\n");
    printf("加速比: %.2fx\n", speedup);
    printf("性能提升: %.1f%%\n", (speedup - 1.0) * 100.0);
    printf("GFLOPS提升: %.2f -> %.2f (+%.1f%%)\n", 
           original_gflops, optimized_gflops, 
           (optimized_gflops - original_gflops) / original_gflops * 100.0);
    
    // 精度验证
    printf("\n=== 精度验证 ===\n");
    g_disable_sd865_optimization = true;
    struct ggml_tensor* result_orig = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph* graph_orig = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_orig, result_orig);
    ggml_graph_compute_with_ctx(ctx, graph_orig, 1);
    
    g_disable_sd865_optimization = false;
    struct ggml_tensor* result_opt = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph* graph_opt = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_opt, result_opt);
    ggml_graph_compute_with_ctx(ctx, graph_opt, 1);
    
    float* data_orig = (float*)result_orig->data;
    float* data_opt = (float*)result_opt->data;
    
    double max_diff = 0.0;
    double avg_diff = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = fabs(data_orig[i] - data_opt[i]);
        if (diff > max_diff) max_diff = diff;
        avg_diff += diff;
    }
    avg_diff /= (M * N);
    
    printf("最大误差: %.6e\n", max_diff);
    printf("平均误差: %.6e\n", avg_diff);
    printf("精度验证: %s\n", max_diff < 1e-5 ? "通过" : "失败");
    
    // 清理
    ggml_free(ctx);
    printf("\n测试完成！\n");
}

int main(int argc, char* argv[]) {
    printf("骁龙865 Q8_0×Q8_0 GEMM性能对比测试\n");
    printf("编译时间: %s %s\n", __DATE__, __TIME__);
    
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
    
    printf("✓ SD865优化: %s\n", ggml_sd865_is_supported() ? "支持" : "不支持");
    
    srand(time(NULL));
    
    // 测试不同的矩阵尺寸
    printf("\n开始性能对比测试...\n");
    
    // Llama2-7B典型的GEMM尺寸
    benchmark_q8_0_gemm(4096, 1, 4096, 20);    // 单token推理
    benchmark_q8_0_gemm(4096, 4, 4096, 10);    // 小批量推理
    benchmark_q8_0_gemm(2048, 1, 2048, 30);    // 中等尺寸测试
    
    return 0;
}
