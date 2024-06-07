import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter


x = ops.ones(5, mindspore.float32)  # input tensor
y = ops.zeros(3, mindspore.float32)  # expected output
w = Parameter(Tensor(np.random.randn(5, 3), mindspore.float32), name='w')  # weight
b = Parameter(Tensor(np.random.randn(3,), mindspore.float32), name='b')  # bias

print("w------------")
print(Parameter.asnumpy(w))
print("-------------")

print("b------------")
print(Parameter.asnumpy(b))
print("-------------")


def function(x, y, w, b):
    z = ops.matmul(x, w) + b
    loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
    return loss


# 测试二元交叉熵损失函数
loss = function(x, y, w, b)
print(loss)

grad_fn = mindspore.grad(function, (2, 3))

grads = grad_fn(x, y, w, b)
print(grads)


# 消除张量影响测试
# 注意该函数除了输出与上面定义的原function函数是相同的
print('计算多个参数的导数')


def function_with_logits(x, y, w, b):
    z = ops.matmul(x, w) + b
    loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
    return loss, z


grad_fn = mindspore.grad(function_with_logits, (2, 3))
grads = grad_fn(x, y, w, b)
print(grads)

print("消除部分张量对梯度的影响")


def function_stop_gradient(x, y, w, b):
    z = ops.matmul(x, w) + b
    loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
    return loss, ops.stop_gradient(z)


grad_fn = mindspore.grad(function_stop_gradient, (2, 3))
grads = grad_fn(x, y, w, b)
print(grads)


print('Auxiliary data 辅助数据测试')
grad_fn = mindspore.grad(function_with_logits, (2, 3), has_aux=True)
grads, (z,) = grad_fn(x, y, w, b)
print(grads, z)


print("开始实测网络模型的反向传播")


# 定义神经网络模型
# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = w
        self.b = b

    def construct(self, x):
        z = ops.matmul(x, self.w) + self.b
        return z


# Instantiate model
model = Network()
# Instantiate loss function
loss_fn = nn.BCEWithLogitsLoss()


# Define forward function
def forward_fn(x, y):
    z = model(x)
    loss = loss_fn(z, y)
    return loss


# 注入损失函数
grad_fn = mindspore.value_and_grad(forward_fn, None, weights=model.trainable_params())
loss, grads = grad_fn(x, y)
print(grads)
