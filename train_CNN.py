import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

#首先读取数据
#分别构建训练集和测试机（验证集）
#DataLoader来迭代数据
#定义超参数
input_size=28  #图像的总尺寸28*28
num_classes=10  #标签的种类数
num_epochs=3  #训练的总循环周期
batch_size=64  #一个批次的大小，64张图片

#训练集
train_dataset=datasets.MNIST(root='./data',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)
#测试集
test_dataset=datasets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())
#构建batch数据
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

#卷积网络模块构建
#一般卷积层、relu层、池化层可以写成一个套餐
#注意卷积最后结果还是一个特征图，需要把图转化为向量才能做分类或者回归任务
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(  #输入大小（1,28,28）
            nn.Conv2d(
                in_channels=1,  #灰度图
                out_channels=16,  #要得到多少个特征图
                kernel_size=5,  #卷积核大小
                stride=1,  #步长
                padding=2  #如果希望卷积后大小和原来一样，要设置padding=（kernel_size-1）/2 if stride=1
            ),  #输出的特征图为（16,14,14）
            nn.ReLU(),  #relu层
            nn.MaxPool2d(kernel_size=2),  #进行池化操作（2*2区域），输出结果为（16,14,14）
        )
        self.conv2=nn.Sequential(  #下一个套餐的输入（16,14,14）
            nn.Conv2d(16,32,5,1,2),  #输出（32,14,14）
            nn.ReLU(),  #relu层
            nn.MaxPool2d(2),  #输出（32,7,7）
        )
        self.out=nn.Linear(32*7*7,10)  #全连接层得到的结果

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)  #flatten操作，结果为（hatch_size,32*7*7）
        output=self.out(x)
        return output