from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
import os
import mindspore.dataset.transforms.transforms as C
import mindspore.dataset.vision as CV
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
import mindspore.nn as nn
from mindspore import Model
from mindspore.nn import Accuracy


# 数据处理过程
def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    # 定义数据集
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml * rescale, shift_nml)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)
    # 使用map映射函数，将数据操作应用到数据集
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=[resize_op, rescale_nml_op, hwc2chw_op], input_columns="image",
                            num_parallel_workers=num_parallel_workers)

    # 进行shuffle、batch操作
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    return mnist_ds


class MLPNet(nn.Cell):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Dense(32*32, 128)
        self.dense2 = nn.Dense(128, 64)
        self.dense3 = nn.Dense(64, 10)

    def construct(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = self.dense3(x)
        return logits


net = MLPNet()
# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 定义优化器
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
mnist_path = "MNIST_Data/"
# 定义测试数据集，batch_size设置为1，则取出一张图片
ds_test = create_dataset(os.path.join(mnist_path, "test"), batch_size=1).create_dict_iterator()
data = next(ds_test)
# images为测试图片，labels为测试图片的实际分类
images = data["image"].asnumpy()
labels = data["label"].asnumpy()
# 加载已经保存的用于测试的模型
param_dict = load_checkpoint("./AddtionFromOpenmlsys/checkpoint_lenet-1_1875.ckpt")
# 加载参数到网络中
load_param_into_net(net, param_dict)
# 使用函数model.predict预测image对应分类
output = model.predict(Tensor(data['image']))
# 输出预测分类与实际分类
print(f'Predicted: "{output[0]}", Actual: "{labels[0]}"')
