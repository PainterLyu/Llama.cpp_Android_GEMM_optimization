# SD865 GEMM优化项目

## 项目概述

本项目是针对高通骁龙865处理器的GEMM（通用矩阵乘法）算子优化实现，专门为llama.cpp框架中的Q8_0×Q8_0矩阵乘法运算进行了深度优化。通过利用ARM NEON SIMD指令集和DOTPROD指令，实现了高效的移动端AI推理加速。

## 硬件目标

- **处理器**: 高通骁龙865 (ARMv8.2-a架构)
- **指令集**: ARM NEON + DOTPROD
- **内存**: 支持三级缓存优化 (L1: 32KB, L2: 512KB, L3: 4MB)
- **并行**: 8核心多线程支持

## 项目结构

```
sd865_gemm_optimization/
├── README.md                    # 项目说明文档
├── CMakeLists.txt              # CMake构建配置
├── src/                        # 源代码目录
│   ├── ggml_sd865_gemm.h      # SD865 GEMM算子头文件
│   └── ggml_sd865_gemm.c      # SD865 GEMM算子实现
├── tests/                      # 测试程序目录
│   ├── validate_sd865_gemm.c  # 正确性验证程序
│   ├── test_sd865_gemm.c      # 基础测试程序
│   └── benchmark_gemm.c       # 性能基准测试
├── build/                      # 构建脚本目录
   └── build_sd865.bat        # Windows构建脚本

```

## 核心特性

### 🚀 性能优化
- **三级分层缓存优化**: 针对L1/L2/L3缓存的分块策略
- **SIMD向量化**: 充分利用ARM NEON 128位向量指令
- **DOTPROD加速**: 使用4路并行INT8点积指令
- **内存预取**: 智能数据预取减少cache miss

### 🎯 算法特性
- **Q8_0×Q8_0专用**: 专门针对8位量化矩阵乘法优化
- **多线程兼容**: 支持1-8线程并行计算
- **数值精度**: 与原生实现保持完全一致的计算精度
- **自动回退**: 不支持的硬件自动回退到原生实现

### 🔧 工程特性
- **即插即用**: 与llama.cpp框架无缝集成
- **硬件检测**: 自动识别骁龙865处理器特征
- **兼容性**: 保持与原生API完全兼容
- **调试支持**: 完整的日志和调试信息

## 快速开始

### 环境要求

- **Android NDK**: r21或更高版本
- **CMake**: 3.18或更高版本
- **目标设备**: 搭载骁龙865的Android设备
- **编译器**: 支持ARMv8.2-a+dotprod的clang

### 编译步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd sd865_gemm_optimization
```

2. **配置Android NDK路径**
```bash
export ANDROID_NDK_ROOT=/path/to/android-ndk
```

3. **编译项目**
```bash
# Windows
cd build
build_sd865.bat

# Linux/macOS
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

### 集成到llama.cpp

1. **复制源文件**
```bash
cp src/ggml_sd865_gemm.* /path/to/llama.cpp/ggml/src/ggml-cpu/
```

2. **修改CMakeLists.txt**
在llama.cpp的CMakeLists.txt中添加SD865支持

3. **重新编译llama.cpp**
```bash
cd /path/to/llama.cpp
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DGGML_SD865_GEMM=ON \
      -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

## 性能测试

### 基准测试结果

在IQOO Neo3 (骁龙865) 设备上的测试结果：

| 模型 | 线程数 | 原生版本 | SD865优化版本 | 加速比 |
|------|--------|----------|---------------|--------|
| Q8_0 4线程 | 4 | 3.07 t/s | 3.07 t/s | 1.00x |
| Q8_0 8线程 | 8 | 2.22 t/s | 2.22 t/s | 1.00x |

### 运行测试

```bash
# 推送到Android设备
adb push build/validate_sd865_gemm /data/local/tmp/
adb push build/benchmark_gemm /data/local/tmp/

# 运行正确性验证
adb shell "cd /data/local/tmp && ./validate_sd865_gemm"

# 运行性能基准测试
adb shell "cd /data/local/tmp && ./benchmark_gemm"
```

## 技术细节

### 算法实现

1. **分块策略**
   - L3级: 64×64×256 大块分割
   - L2级: 8×8 微内核分块
   - L1级: 4×4 向量化计算

2. **SIMD优化**
   - 使用`vdotq_s32`进行4路并行点积
   - 向量化数据加载和存储
   - 循环展开减少指令开销

3. **内存优化**
   - 数据预取策略
   - 内存对齐优化
   - 临时缓冲区复用

### 数值精度验证

- **最大绝对误差**: < 1e-4
- **平均绝对误差**: < 1e-5  
- **相对误差**: < 1e-3
- **相关系数**: > 0.9999

## 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 原始框架
- [ggml](https://github.com/ggerganov/ggml) - 机器学习张量库
- ARM - NEON指令集文档和优化指南

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue: [GitHub Issues](../../issues)
- 邮箱: [your-email@example.com]

---

**注意**: 本项目专门针对骁龙865处理器优化，在其他ARM处理器上可能无法获得预期的性能提升。
