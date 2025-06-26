# 构建说明

## 环境准备

### 必需工具
- **Android NDK**: r21或更高版本
- **CMake**: 3.18或更高版本  
- **Git**: 用于版本控制
- **ADB**: 用于设备调试

### 目标设备要求
- **处理器**: 高通骁龙865
- **架构**: ARM64 (aarch64)
- **Android版本**: 7.0 (API 24) 或更高
- **指令集支持**: NEON + DOTPROD

## 构建步骤

### 1. 克隆项目
```bash
git clone <your-repository-url>
cd sd865_gemm_optimization
```

### 2. 设置环境变量
```bash
# Windows
set ANDROID_NDK_ROOT=E:\path\to\android-ndk-r28b

# Linux/macOS
export ANDROID_NDK_ROOT=/path/to/android-ndk-r28b
```

### 3. 创建构建目录
```bash
mkdir build
cd build
```

### 4. 配置CMake
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK_ROOT%/build/cmake/android.toolchain.cmake ^
      -DANDROID_ABI=arm64-v8a ^
      -DANDROID_PLATFORM=android-28 ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod -O3" ^
      ..
```

### 5. 编译项目
```bash
# Windows
cmake --build . --config Release

# Linux/macOS  
make -j8
```

## 测试验证

### 1. 推送到设备
```bash
adb push validate_sd865_gemm /data/local/tmp/
adb push test_sd865_gemm /data/local/tmp/
adb push benchmark_gemm /data/local/tmp/
```

### 2. 设置执行权限
```bash
adb shell "chmod +x /data/local/tmp/validate_sd865_gemm"
adb shell "chmod +x /data/local/tmp/test_sd865_gemm"
adb shell "chmod +x /data/local/tmp/benchmark_gemm"
```

### 3. 运行测试
```bash
# 正确性验证
adb shell "/data/local/tmp/validate_sd865_gemm"

# 性能基准测试
adb shell "/data/local/tmp/benchmark_gemm"

# 基础功能测试
adb shell "/data/local/tmp/test_sd865_gemm"
```

## 集成到llama.cpp

### 1. 复制源文件
```bash
cp src/ggml_sd865_gemm.h /path/to/llama.cpp/ggml/src/ggml-cpu/
cp src/ggml_sd865_gemm.c /path/to/llama.cpp/ggml/src/ggml-cpu/
```

### 2. 修改CMakeLists.txt
在llama.cpp的`ggml/src/ggml-cpu/CMakeLists.txt`中添加：
```cmake
# SD865 GEMM optimization
option(GGML_SD865_GEMM "Enable SD865 GEMM optimization" OFF)

if(GGML_SD865_GEMM)
    target_sources(ggml-cpu PRIVATE ggml_sd865_gemm.c)
    target_compile_definitions(ggml-cpu PRIVATE GGML_SD865_GEMM)
endif()
```

### 3. 重新编译llama.cpp
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

## 故障排除

### 常见问题

1. **编译错误: 找不到DOTPROD指令**
   - 确保使用`-march=armv8.2-a+dotprod`编译选项
   - 检查NDK版本是否支持ARMv8.2-a

2. **运行时错误: 硬件检测失败**
   - 确认设备确实是骁龙865处理器
   - 检查`/proc/cpuinfo`确认DOTPROD支持

3. **性能没有提升**
   - 确认优化版本被正确调用
   - 检查编译优化选项是否正确
   - 验证测试矩阵大小是否适合优化

4. **数值精度问题**
   - 运行完整的正确性验证测试
   - 检查量化和反量化过程
   - 确认SIMD指令使用正确

### 调试技巧

1. **启用详细日志**
```c
#define GGML_SD865_DEBUG 1  // 在编译时定义
```

2. **使用GDB调试**
```bash
adb shell gdbserver :5039 /data/local/tmp/test_sd865_gemm
# 在另一个终端
adb forward tcp:5039 tcp:5039
gdb-multiarch
(gdb) target remote :5039
```

3. **性能分析**
```bash
# 使用simpleperf进行性能分析
adb shell simpleperf record -o /data/local/tmp/perf.data /data/local/tmp/benchmark_gemm
adb shell simpleperf report -i /data/local/tmp/perf.data
```

## 贡献代码

在提交代码前，请确保：

1. 所有测试通过
2. 代码符合项目编码规范
3. 添加了适当的注释和文档
4. 性能测试结果符合预期

```bash
# 运行完整测试套件
./run_all_tests.sh

# 检查代码格式
clang-format -i src/*.c src/*.h
```
