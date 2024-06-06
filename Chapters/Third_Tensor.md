## 张量 Tensor

​	张量（Tensor）是一个可用来表示在一些矢量、标量和其他张量之间的线性关系的多线性函数，这些线性关系的基本例子有内积、外积、线性映射以及笛卡儿积。其坐标在 $n$ 维空间内，有 $n^r$ 个分量的一种量，其中每个分量都是坐标的函数，而在坐标变换时，这些分量也依照某些规则作线性变换。$r$ 称为该张量的秩或阶（与矩阵的秩和阶均无关系）。

​	张量是一种特殊的数据结构，与数组和矩阵非常相似，他表示的是一种多维的“矩阵”的集合。张量（[Tensor](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore/mindspore.Tensor.html)）是MindSpore网络运算中的基本数据结构，本教程主要介绍张量和稀疏张量的属性及用法。

​	*矩阵的秩或阶是人工智能中基础且常考的考点：一般形式是求矩阵秩是多少*

下面是对张量在MindSpore中的实践

### 创建张量

1. **根据数据直接生成**

```python
data = [1, 0, 1, 0]
x_data = Tensor(data)
print(x_data, x_data.shape, x_data.dtype)
```

> [1 0 1 0] (4,) Int64

2. **从NumPy数组生成**

```python
np_array = np.array(data)
x_np = Tensor(np_array)
print(x_np, x_np.shape, x_np.dtype)
```

> [1 0 1 0] (4,) Int64

3. **使用init初始化器构造张量**

​	当使用`init`初始化器对张量进行初始化时，支持传入的参数有`init`、`shape`、`dtype`。

   - `init`: 支持传入[initializer](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore.common.initializer.html)的子类。如：下方示例中的 [One()](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore.common.initializer.html#mindspore.common.initializer.One) 和 [Normal()](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Normal)。

  - `shape`: 支持传入 `list`、`tuple`、 `int`。

  - `dtype`: 支持传入[mindspore.dtype](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore/mindspore.dtype.html#mindspore.dtype)。

4. 继承张量并形成新的张量

```python
from mindspore import ops

x_ones = ops.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_zeros = ops.zeros_like(x_data)
print(f"Zeros Tensor: \n {x_zeros} \n")
```

### 张量的属性

- 张量的属性包括形状、数据类型、转置张量、单个元素大小、占用字节数量、维数、元素个数和每一维步长。
  - 形状（shape）：`Tensor`的shape，是一个tuple（元组，python中的数据类型标签）。
  - 数据类型（dtype）：`Tensor`的dtype，是MindSpore的一个数据类型。
  - 单个元素大小（itemsize）： `Tensor`中每一个元素占用字节数，是一个整数。
  - 占用字节数量（nbytes）： `Tensor`占用的总字节数，是一个整数。
  - 维数（ndim）： `Tensor`的秩，也就是len(tensor.shape)，是一个整数。
  - 元素个数（size）： `Tensor`中所有元素的个数，是一个整数。
  - 每一维步长（strides）： `Tensor`每一维所需要的字节数，是一个tuple。

为更简单的理解shape的含义，我修改了一下官方文档中的x张量

```python
x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.int32)

print("x_shape:", x.shape)
print("x_dtype:", x.dtype)
print("x_itemsize:", x.itemsize)
print("x_nbytes:", x.nbytes)
print("x_ndim:", x.ndim)
print("x_size:", x.size)
print("x_strides:", x.strides)
```

> x_shape: (3, 2)
> x_dtype: Int32
> x_itemsize: 4
> x_nbytes: 24
> x_ndim: 2
> x_size: 6
> x_strides: (8, 4)

### 张量的下标索引

Tensor索引与Numpy索引类似，索引从0开始编制，负索引表示按倒序编制，冒号`:`和 `...`用于对数据进行切片。切片的意思是后面的参数是按行算的还是按列算的，详细请看代码

```
tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))

print("First row: {}".format(tensor[0]))
print("value of bottom right corner: {}".format(tensor[1, 1]))
print("Last column: {}".format(tensor[:, -1]))
print("First column: {}".format(tensor[..., 0]))
```

> First row: [0. 1.]
> value of bottom right corner: 3.0
> Last column: [1. 3.]
> First column: [0. 2.]

### 张量运算

张量之间有很多运算，包括算术、线性代数、矩阵处理（转置、标引、切片）、采样等，张量运算和NumPy的使用方式类似，下面介绍其中几种操作。

> 普通算术运算有：加（+）、减（-）、乘（*）、除（/）、取模（%）、整除（//）。

```python
x = Tensor(np.array([1, 2, 3]), mindspore.float32)
y = Tensor(np.array([4, 5, 6]), mindspore.float32)

output_add = x + y
output_sub = x - y
output_mul = x * y
output_div = y / x
output_mod = y % x
output_floordiv = y // x

print("add:", output_add)
print("sub:", output_sub)
print("mul:", output_mul)
print("div:", output_div)
print("mod:", output_mod)
print("floordiv:", output_floordiv)
```

> add: [5. 7. 9.]
> sub: [-3. -3. -3.]
> mul: [ 4. 10. 18.]
> div: [4.  2.5 2. ]
> mod: [0. 1. 0.]
> floordiv: [4. 2. 2.]

对于一些函数的使用，这里之贴出定义，详细运行库中代码（个人觉得没必要，知道这些函数即可，毕竟到处可见类似函数）

[concat](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.concat.html)() : 将给定维度上的一系列张量连接起来，0表示最高得也就是直接通过`张量名[下标索引]`时的张量名所代表的元组。

[stack](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.stack.html)()：则是从另一个维度上将两个张量合并起来。（新建一个维度）
