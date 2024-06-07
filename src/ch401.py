# 引包
import numpy as np
from mindspore.dataset import vision
from mindspore.dataset import MnistDataset, GeneratorDataset
import matplotlib.pyplot as plt

# 下载完成后注释
# # Download data from open datasets
# from download import download

# url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
#     "notebook/datasets/MNIST_Data.zip"
# path = download(url, "./", kind="zip", replace=True)

train_dataset = MnistDataset("MNIST_Data/train", shuffle=False)
print(type(train_dataset))


# 定义图片训练数据可视化函数
def visualize(dataset):
    figure = plt.figure(figsize=(4, 4))
    cols, rows = 3, 3

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for idx, (image, label) in enumerate(dataset.create_tuple_iterator()):
        figure.add_subplot(rows, cols, idx + 1)
        plt.title(int(label))
        plt.axis("off")
        plt.imshow(image.asnumpy().squeeze(), cmap="gray")
        if idx == cols * rows - 1:
            break
    plt.show()


# 初始调用可视化函数
# visualize(train_dataset)

# shuffle操作观察
# train_dataset = train_dataset.shuffle(buffer_size=64)
# visualize(train_dataset)

# map 操作观察
image, label = next(train_dataset.create_tuple_iterator())
print("map前：")
print(image.shape, image.dtype)
train_dataset = train_dataset.map(vision.Rescale(1.0 / 255.0, 0), input_columns='image')
image, label = next(train_dataset.create_tuple_iterator())
print("map后：")
print(image.shape, image.dtype)

print("----------------")
print()

# 一般设定固定batchSize
train_dataset = train_dataset.batch(batch_size=32)
# batch后的数据增加一维，大小为batch_size
image, label = next(train_dataset.create_tuple_iterator())
print(image.shape, image.dtype)

print("----------------")
print()
print("可随机访问数据集")


# Random-accessible object as input source
class RandomAccessDataset:
    def __init__(self):
        self._data = np.ones((5, 2))
        self._label = np.zeros((5, 1))

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)


loader = RandomAccessDataset()
dataset = GeneratorDataset(source=loader, column_names=["data", "label"])

for data in dataset:
    print(data)

# list, tuple are also supported.
loader = [np.array(0), np.array(1), np.array(2)]
dataset = GeneratorDataset(source=loader, column_names=["data"])

for data in dataset:
    print(data)

print("----------------")
print()
print("可迭代数据集")


# Iterator as input source
class IterableDataset():
    def __init__(self, start, end):
        '''init the class object to hold the data'''
        self.start = start
        self.end = end

    def __next__(self):
        '''iter one data and return'''
        return next(self.data)

    def __iter__(self):
        '''reset the iter'''
        self.data = iter(range(self.start, self.end))
        return self


loader = IterableDataset(1, 5)
dataset = GeneratorDataset(source=loader, column_names=["data"])

for d in dataset:
    print(d)

print("----------------")
print()
print("生成器")


# Generator
def my_generator(start, end):
    for i in range(start, end):
        yield i


# since a generator instance can be only iterated once, we need to wrap it by lambda to generate multiple instances
dataset = GeneratorDataset(source=my_generator(3, 6), column_names=["data"])

for d in dataset:
    print(d)
