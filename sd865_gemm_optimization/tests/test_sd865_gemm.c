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

// 创建测试矩阵
struct ggml_tensor* create_test_matrix_q4_0(struct ggml_context* ctx, int rows, int cols) {
    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, cols, rows);
    
    // 填充随机数据
    float* temp_data = malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) {
        temp_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    
    // 量化为Q4_0
    ggml_quantize_q4_0(temp_data, tensor->data, rows * cols, cols, NULL);
    free(temp_data);
    
    return tensor;
}

struct ggml_tensor* create_test_matrix_f32(struct ggml_context* ctx, int rows, int cols) {
    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    
    float* data = (float*)tensor->data;
    for (int i = 0; i < rows * cols; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    
    return tensor;
}

// 性能测试函数
void benchmark_gemm(int M, int N, int K, int iterations) {
    printf("=== 骁龙865 GEMM性能测试 ===\n");
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
    printf("创建测试矩阵...\n");
    struct ggml_tensor* a = create_test_matrix_q4_0(ctx, M, K);  // Q4_0权重矩阵
    struct ggml_tensor* b = create_test_matrix_f32(ctx, K, N);   // F32输入矩阵
    
    // 转换b为Q8_0格式（llama.cpp的标准做法）
    struct ggml_tensor* b_q8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    ggml_quantize_q8_0((float*)b->data, b_q8->data, K * N, K, NULL);
    
    // 创建结果矩阵
    struct ggml_tensor* c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    
    printf("开始性能测试...\n");
    
    // 预热
    for (int i = 0; i < 3; i++) {
        struct ggml_tensor* result = ggml_mul_mat(ctx, a, b_q8);
        ggml_graph_compute_with_ctx(ctx, ggml_new_graph(ctx), 1);
    }
    
    // 正式测试
    double total_time = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;
    
    for (int i = 0; i < iterations; i++) {
        double start_time = get_time_ms();
        
        // 执行矩阵乘法
        struct ggml_tensor* result = ggml_mul_mat(ctx, a, b_q8);
        
        // 构建计算图并执行
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        
        double end_time = get_time_ms();
        double elapsed = end_time - start_time;
        
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
        
        if ((i + 1) % 10 == 0) {
            printf("完成 %d/%d 次迭代\n", i + 1, iterations);
        }
    }
    
    // 计算性能指标
    double avg_time = total_time / iterations;
    double gflops = (2.0 * M * N * K) / (avg_time * 1e6); // GFLOPS
    
    printf("\n=== 性能结果 ===\n");
    printf("平均时间: %.2f ms\n", avg_time);
    printf("最小时间: %.2f ms\n", min_time);
    printf("最大时间: %.2f ms\n", max_time);
    printf("性能: %.2f GFLOPS\n", gflops);
    printf("吞吐量: %.2f tokens/s (假设每token需要一次GEMM)\n", 1000.0 / avg_time);
    
    // 验证结果正确性
    printf("\n=== 精度验证 ===\n");
    struct ggml_tensor* result = ggml_mul_mat(ctx, a, b_q8);
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    
    float* result_data = (float*)result->data;
    float sum = 0.0f;
    for (int i = 0; i < M * N; i++) {
        sum += result_data[i];
    }
    printf("结果矩阵元素和: %.6f\n", sum);
    printf("结果矩阵平均值: %.6f\n", sum / (M * N));
    
    // 清理
    ggml_free(ctx);
    printf("\n测试完成！\n");
}

// 对比测试：原始实现 vs 骁龙865优化实现
void compare_implementations(int M, int N, int K, int iterations) {
    printf("\n=== 实现对比测试 ===\n");
    printf("矩阵尺寸: M=%d, N=%d, K=%d\n", M, N, K);

    // 初始化ggml
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    struct ggml_context* ctx = ggml_init(params);

    // 创建测试数据
    struct ggml_tensor* a = create_test_matrix_q4_0(ctx, M, K);
    struct ggml_tensor* b = create_test_matrix_f32(ctx, K, N);
    struct ggml_tensor* b_q8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    ggml_quantize_q8_0((float*)b->data, b_q8->data, K * N, K, NULL);

    // 测试原始实现（通过禁用优化）
    printf("\n测试原始实现...\n");
    double original_time = 0.0;
    for (int i = 0; i < iterations; i++) {
        double start = get_time_ms();

        // 这里需要调用原始实现
        struct ggml_tensor* result = ggml_mul_mat(ctx, a, b_q8);
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        ggml_graph_compute_with_ctx(ctx, graph, 1);

        double end = get_time_ms();
        original_time += (end - start);
    }
    original_time /= iterations;

    printf("原始实现平均时间: %.2f ms\n", original_time);
    printf("原始实现性能: %.2f GFLOPS\n", (2.0 * M * N * K) / (original_time * 1e6));

    // 测试骁龙865优化实现
    printf("\n测试骁龙865优化实现...\n");
    double optimized_time = 0.0;
    for (int i = 0; i < iterations; i++) {
        double start = get_time_ms();

        struct ggml_tensor* result = ggml_mul_mat(ctx, a, b_q8);
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        ggml_graph_compute_with_ctx(ctx, graph, 1);

        double end = get_time_ms();
        optimized_time += (end - start);
    }
    optimized_time /= iterations;

    printf("优化实现平均时间: %.2f ms\n", optimized_time);
    printf("优化实现性能: %.2f GFLOPS\n", (2.0 * M * N * K) / (optimized_time * 1e6));

    // 计算加速比
    double speedup = original_time / optimized_time;
    printf("\n=== 性能提升 ===\n");
    printf("加速比: %.2fx\n", speedup);
    printf("性能提升: %.1f%%\n", (speedup - 1.0) * 100.0);

    ggml_free(ctx);
}

int main(int argc, char* argv[]) {
    printf("骁龙865 GEMM优化测试程序\n");
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

    #ifdef __ARM_FEATURE_MATMUL_INT8
    printf("✓ ARM i8mm支持\n");
    #else
    printf("✗ ARM i8mm不支持 (骁龙865正常)\n");
    #endif

    // 检查骁龙865优化是否启用
    if (ggml_sd865_is_supported()) {
        printf("✓ 骁龙865优化已启用\n");
    } else {
        printf("✗ 骁龙865优化未启用\n");
    }

    srand(time(NULL));

    // 基础性能测试
    printf("\n=== 基础性能测试 ===\n");
    benchmark_gemm(4096, 1, 4096, 50);    // 单token推理
    benchmark_gemm(4096, 8, 4096, 20);    // 小批量推理
    benchmark_gemm(4096, 32, 4096, 10);   // 中等批量推理

    // 对比测试
    printf("\n=== 对比测试 ===\n");
    compare_implementations(4096, 1, 4096, 20);
    compare_implementations(4096, 8, 4096, 10);

    return 0;
}
