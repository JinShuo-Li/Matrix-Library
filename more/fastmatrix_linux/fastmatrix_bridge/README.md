# fastmatrix_bridge

这是一个“**Python 接口 + C++ 后端 + 多线程优化**”的矩阵库样例工程。

## 设计目标

参考一般 Matrix-Library 项目的功能风格，但把职责拆成两层：

- **Python 层**：负责友好语法、运算符重载、易用接口
- **C++ 层**：负责高性能数值计算和多线程执行

## 当前实现

### Python 层

- `Matrix` 类
- 支持 `@` 矩阵乘法
- 支持 `+` / `-`
- 支持 `.T` 转置
- 支持 `.numpy()` 导出为 NumPy 数组

### C++ 层

- `fm_add`
- `fm_sub`
- `fm_transpose`
- `fm_matmul`

## 矩阵乘法优化策略

1. **先把 B 转置**，把列访问变成连续内存访问
2. **blocking 分块**，提升缓存命中率
3. **按行切块多线程并行**，减少线程间写冲突
4. 默认线程数为 `hardware_concurrency()`

## 编译

```bash
bash build.sh
```

## 使用

```python
from fastmatrix import Matrix

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

C = A @ B
print(C.numpy())
print(C.T.numpy())
```

## 自定义线程和块大小

```python
C = A.matmul(B, threads=8, block_size=64)
```

## 目录结构

```text
fastmatrix_bridge/
├── cpp/
│   └── fastmatrix.cpp
├── fastmatrix/
│   ├── __init__.py
│   ├── backend.py
│   ├── matrix.py
│   └── libfastmatrix.so / dylib / dll
├── build.sh
├── example.py
└── benchmark.py
```
