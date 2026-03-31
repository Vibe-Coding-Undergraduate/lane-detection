import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision

'''
运行说明
安装依赖命令，要安装这个版本。
pip install torch==1.4.0 torchvision==0.5.0
pip install matplotlib numpy
代码按照内容的顺序，方便阅读。
'''
# 1.1.2  导入数据集

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', #root表示数据加载的相对目录
                   train=True, #train表示是否加载数据库的训练集，False时加载测试集
                   download=True,#download表示是否自动下载
                   transform=transforms.Compose([#transform表示对数据进行预处理的操作
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),batch_size=64, shuffle=True)#batch_size表示该批次的数据量  shuffle表示是否洗牌
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),batch_size=64, shuffle=True)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# 得到batch中的数据
dataiter = iter(train_loader)
images, labels = next(dataiter)
# 展示图片
imshow(torchvision.utils.make_grid(images))


# 1.2.1  定义神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F#可以调用一些常见的函数，例如非线性以及池化等

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # 输入图片是1 channel输出是6 channel 利用5x5的核大小
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接 从16 * 4 * 4的维度转成120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # 在(2, 2)的窗口上进行池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)#(2,2)也可以直接写成数字2
        x = x.view(-1, self.num_flat_features(x))#将维度转成以batch为第一维 剩余维数相乘为第二维
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # 第一个维度batch不考虑
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# 1.2.2  前向传播

image = images[:2]
label = labels[:2]
print(image.size())
print(label)
out = net(image)
print(out)

# 1.2.3  计算损失

image = images[:2]
label = labels[:2]
out = net(image)
criterion = nn.CrossEntropyLoss()
loss = criterion(out, label)
print(loss)

# 1.2.4  反向传播与更新参数

#创建优化器
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)#lr代表学习率
criterion = nn.CrossEntropyLoss()
# 在训练过程中
image = images[:2]
label = labels[:2]
optimizer.zero_grad()   # 消除累积梯度
out = net(image)
loss = criterion(out, label)
loss.backward()
optimizer.step()    # 更新参数

# 1.3  开始训练

def train(epoch):
    net.train() # 设置为training模式
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # 得到输入 和 标签
        inputs, labels = data
        # 消除梯度
        optimizer.zero_grad()
        # 前向传播 计算损失 后向传播 更新参数
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 打印日志
        running_loss += loss.item()
        if i % 100 == 0:    # 每100个batch打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

train(1)

# 1.4  观察模型预测效果

correct = 0
total = 0
with torch.no_grad():#或者model.eval()
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
