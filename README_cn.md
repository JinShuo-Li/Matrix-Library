# MatLibrary 中文文档

## 1. 项目概述

`matlibrary.py` 是一个统一的教学型矩阵库，核心为单一 `Matrix` 类。

它将多个历史版本的实现重构为统一架构，目标是：

- 保持 `data/dim` 风格的数据组织。
- 同时支持精确有理数与浮点数值计算。
- 覆盖基础矩阵运算到高级分解算法。
- 在同类问题中提供算法对比与推荐。

该库定位于**学习、算法理解、课程实验**，不是面向大规模高性能生产环境。

---

## 2. 设计目标

1. **单文件统一实现**
   - 把分散在不同文件的功能收敛到 `matlibrary.py`。
2. **架构一致性**
   - 保留 `matrix.py` 风格的 `data/dim` 主结构。
3. **精确与数值并存**
   - `dtype="fraction"`：适合代数精确计算。
   - `dtype="float"`：适合分解、迭代、正交化等数值算法。
4. **同题多法可比较**
   - 在 QR、特征值、SVD 等问题上保留不同方法并给出建议。
5. **可读可解释**
   - 实现尽量贴近线性代数教材公式。

---

## 3. 整体架构

## 3.1 数据模型

每个矩阵对象包含：

- `self.data`：嵌套列表。
- `self.dim`：`(行数, 列数)`。
- `self.dtype`：`"fraction"` 或 `"float"`。
- `self.tolerance`：数值近零阈值。

构造函数：

```python
Matrix(data=None, dim=None, init_value=0, tolerance=1e-10, dtype="fraction")
```

支持三种创建方式：

- 由二维列表 `data` 创建。
- 由 `dim=(m,n)` 与初始值创建。
- 由已有 `Matrix` 深拷贝创建。

## 3.2 双数值模式

### 有理数精确模式 (`dtype="fraction"`)

- 内部元素转为 `Fraction`。
- 适合精确验证与代数推导。
- 在加减乘、消元、行列式、秩、部分代数流程中优势明显。

### 浮点数值模式 (`dtype="float"`)

- 内部元素转为 `float`。
- 适合需要开方、迭代、正交变换的算法：
  - Cholesky
  - QR
  - Jacobi 特征值
  - SVD

## 3.3 内部工具层

关键内部方法：

- `_cast_value`：按 dtype 转换元素。
- `_near_zero`：统一近零判断。
- `_dot`, `_norm`, `_normalize_vec`：向量运算。
- `_mat_vec_mul`：矩阵向量乘法。
- `_complete_orthonormal_basis`：SVD 中补齐正交基。
- `_is_symmetric`：对称性检测。
- `to_float_data`：转换为浮点二维列表。

---

## 4. 方法体系与数学原理

## 4.1 基础代数操作

- `__add__`, `__sub__`：同型矩阵逐元素运算。
- `__mul__`：
  - 支持标量乘法，
  - 支持矩阵乘法。
- `__matmul__`：矩阵乘法别名。
- `__pow__`：平方矩阵快速幂（重复平方法）。

## 4.2 索引与切片

- `__getitem__` 支持 `(int/slice, int/slice)`。
- `__setitem__` 支持标量写入与子矩阵写入（含形状检查）。

## 4.3 高斯消元

方法：

```python
gauss_elimination(pivoting=True)
```

返回：

- 行阶梯形矩阵
- 行交换次数

原理：

- 用初等行变换把矩阵化到行阶梯形。
- 开启部分主元选取（pivoting）后，数值稳定性更好。

## 4.4 行列式

方法：

```python
determinant()
```

原理：

- 消元后取对角线乘积。
- 再乘上 $(-1)^{\text{swap\_count}}$ 修正交换符号。

## 4.5 逆矩阵（Gauss-Jordan）

方法：

```python
inverse()
```

流程：

1. 构造增广矩阵 `[A | I]`。
2. 行变换把左边化成单位阵。
3. 右边即为 $A^{-1}$。

## 4.6 秩

方法：

```python
rank()
```

原理：

- 行阶梯化后统计非零行数（float 模式结合容差判断）。

## 4.7 LU 分解（Doolittle）

方法：

```python
lu_decomposition(pivoting=True)
```

返回：

- `pivoting=True`：`(P, L, U)`，满足 $PA=LU$。
- `pivoting=False`：`(L, U)`。

比较：

- 带主元 LU 更鲁棒，是默认推荐。

## 4.8 Cholesky 分解

方法：

```python
cholesky_decomposition() -> (L, L.T)
```

适用条件：

- 方阵、对称、正定（SPD）。

数学结论：

$$A = LL^T$$

## 4.9 Gram-Schmidt 与 QR 分解

方法：

- `gram_schmidt(method="mgs")`
- `qr_decomposition(method="householder")`

支持方式：

1. Householder（推荐）
2. Modified Gram-Schmidt（MGS）
3. Classical Gram-Schmidt（CGS）

比较：

- Householder 数值稳定性通常最好。
- MGS 通常优于 CGS。

## 4.10 特征值方法

### Jacobi（对称矩阵）

方法：

```python
eigenpairs_jacobi(max_iterations=1000, sort=True)
```

原理：

- 反复做平面旋转，将非对角元消为 0。
- 对实对称矩阵可收敛到对角阵。
- 对角元素为特征值，累计旋转矩阵给出特征向量。

### QR 迭代（一般方阵）

方法：

```python
eigenvalues_qr(iterations=100, qr_method="householder")
```

原理：

- 迭代 $A_{k}=R_kQ_k$，其中 $A_{k-1}=Q_kR_k$。
- 对许多矩阵，对角线逐步逼近特征值。

### 统一接口

- `eigenpairs(method="auto", ...)`
- `eigenvalues(method="auto", ...)`

`auto` 逻辑：

- 若矩阵对称，优先用 Jacobi。
- 否则用 QR 迭代。

## 4.11 奇异值分解 SVD

方法：

```python
singular_value_decomposition() -> (U, Sigma, V_T)
```

实现路线：

1. 计算 $A^TA$。
2. 对 $A^TA$ 做 Jacobi 特征分解，得到 $V$ 与特征值。
3. 奇异值为 $\sigma_i=\sqrt{\lambda_i}$。
4. 用
   $$u_i=\frac{Av_i}{\sigma_i}$$
   构造左奇异向量。
5. 对秩亏情况补齐正交基。
6. 组装完整 $U,\Sigma,V^T$。

为什么这样做：

- 比“独立分解 $AA^T$ 再对齐”更容易保持列对应关系稳定。

## 4.12 Jordan 标准形

方法：

```python
jordan_normal_form()
```

流程：

1. 用 Faddeev-LeVerrier 求特征多项式系数。
2. 用有理根定理找有理特征值。
3. 通过 $(A-\lambda I)^k$ 的秩序列推 Jordan 块大小。
4. 构造 Jordan 标准形。

限制：

- 当前实现是有理特征值场景的精确路径。
- 非有理特征值会抛出可解释异常。

---

## 5. 接口总览

## 5.1 构造与转换

- `Matrix(data=None, dim=None, init_value=0, tolerance=1e-10, dtype="fraction")`
- `to_float_data()`
- `copy()`
- `shape()`
- `reshape(newdim)`

## 5.2 结构与工具

- `transpose()`
- `T()`
- `sum(axis=None)`
- `concatenate(other, index=0)`
- `kronecker_product(other)`
- `Kronecker_product(other)`
- `identity(n, tolerance=1e-10, dtype="fraction")`

## 5.3 运算符

- `__add__`, `__sub__`, `__mul__`, `__rmul__`, `__matmul__`, `__pow__`
- `num_product`
- `__getitem__`, `__setitem__`, `__len__`, `__str__`

## 5.4 核心线性代数

- `gauss_elimination(pivoting=True)`
- `determinant()` / `det()`
- `inverse()`
- `rank()`

## 5.5 分解类接口

- `lu_decomposition(pivoting=True)`
- `cholesky_decomposition()`
- `gram_schmidt(method="mgs")`
- `qr_decomposition(method="householder")`
- `QR_decomposition(method="householder")`

## 5.6 特征值 / SVD / Jordan

- `eigenpairs_jacobi(max_iterations=1000, sort=True)`
- `eigenvalues_qr(iterations=100, qr_method="householder")`
- `eigenpairs(method="auto", ...)`
- `eigenvalues(method="auto", ...)`
- `singular_value_decomposition()`
- `svd()`
- `jordan_normal_form()`

## 5.7 全局辅助函数

- `I(n, tolerance=1e-10, dtype="fraction")`
- `concatenate(items, axis=0)`

---

## 6. 同题多法比较

可以直接调用：

```python
Matrix.method_comparison()
```

会返回 markdown 文本，包含算法对比结论。

推荐默认策略：

- 消元：带主元。
- QR：Householder。
- 特征值：对称走 Jacobi，非对称走 QR。
- SVD：以 $A^TA$ 路线构造。

---

## 7. 复杂度（高层）

以 $n\times n$ 稠密矩阵为参考：

- 矩阵乘法：$O(n^3)$
- 消元/行列式/逆/LU/Cholesky/QR：典型 $O(n^3)$
- Jacobi/QR 特征值：与迭代次数有关，单轮通常立方级
- SVD：由特征分解与向量构造主导

---

## 8. 使用示例

## 8.1 基础运算

```python
from matlibrary import Matrix

A = Matrix(data=[[1, 2], [3, 4]], dtype="fraction")
B = Matrix(data=[[5, 6], [7, 8]], dtype="fraction")

print(A + B)
print(A * B)
```

## 8.2 LU 与逆矩阵

```python
A = Matrix(data=[[0, 2], [1, 3]], dtype="fraction")
P, L, U = A.lu_decomposition(pivoting=True)
A_inv = A.inverse()
```

## 8.3 Cholesky

```python
A = Matrix(data=[[4, 2], [2, 3]], dtype="float")
L, LT = A.cholesky_decomposition()
```

## 8.4 特征值

```python
A = Matrix(data=[[2, 1], [1, 2]], dtype="float")
vals, vecs = A.eigenpairs(method="jacobi")

B = Matrix(data=[[1, 2], [3, 4]], dtype="float")
vals_general = B.eigenvalues(method="qr", iterations=200)
```

## 8.5 SVD

```python
A = Matrix(data=[[3, 2, 2], [2, 3, -2]], dtype="float")
U, Sigma, V_T = A.svd()
A_recon = U * Sigma * V_T
```

---

## 9. 精度与模式建议

1. 作业推导/精确验证优先 `dtype="fraction"`。
2. 分解与迭代算法优先 `dtype="float"`。
3. 浮点流程要设置合理 `tolerance`（如 `1e-10` 到 `1e-12`）。
4. 病态矩阵中，任何实现都可能出现数值敏感。

---

## 10. 当前限制

1. 纯 Python 实现，非高性能版本。
2. Jordan 精确实现仅覆盖有理特征值路径。
3. QR 特征值是教学型无位移迭代。
4. SVD 为教学实现，不等同于 LAPACK 工业级鲁棒性。

---

## 11. 可扩展方向

1. 增加带位移 QR / 隐式 QR。
2. 增加基于双对角化的 SVD。
3. 增加稀疏矩阵支持。
4. 增加范数、条件数等数值分析接口。
5. 增加系统化单元测试。

---

## 12. 快速接口速查

- 精确模式：`Matrix(data=..., dtype="fraction")`
- 数值模式：`Matrix(data=..., dtype="float")`
- 主要分解：
  - `lu_decomposition`
  - `cholesky_decomposition`
  - `qr_decomposition`
  - `eigenpairs`
  - `svd`
  - `jordan_normal_form`
- 全局函数：`I`, `concatenate`

这份文档既可作为使用手册，也可作为线性代数算法学习索引。