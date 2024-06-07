# 引包
import mindspore
from mindspore import nn, ops
from mindspore import Tensor


class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros")
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits


model = Network()
print(model)

X = ops.ones((1, 28, 28), mindspore.float32)
logits = model(X)
# print logits

print(Tensor(logits))
pred_probab = nn.Softmax(axis=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

input_image = ops.ones((3, 28, 28), mindspore.float32)
print(input_image.shape)

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.shape)

layer1 = nn.Dense(in_channels=28*28, out_channels=20)
hidden1 = layer1(flat_image)
print(hidden1.shape)


print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

seq_modules = nn.SequentialCell(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Dense(20, 10)
)

logits = seq_modules(input_image)
print(logits.shape)
