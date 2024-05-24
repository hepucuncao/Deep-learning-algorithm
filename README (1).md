# DeepLearning

2024年4月1日**更新**

在此教程中，我们将对深度学习有一个基本的认识，并介绍几种常用的模型及算法，并对几个经典的模型及算法进行简单的代码实现。


# 目录  
[环境搭建](#环境搭建)

[学习路径](#学习路径)

[基本介绍](#基本介绍)  
- [何为深度学习](#何为深度学习)
- [神经网络](#神经网络)
- [项目结构](#项目结构)

[常用模型及算法](#常用模型及算法)
- [多层感知机(MLP)](#多层感知机(MLP))
- [卷积神经网络(CNN)](#卷积神经网络(CNN))
- [循环神经网络(RNN)](#循环神经网络(RNN))
- [LSTM长短期记忆神经网络](#LSTM长短期记忆神经网络)
- [强化学习(RL)](#强化学习(RL))

***
<a name="环境搭建"></a>
## 环境搭建

在正式开始学习深度学习之前，我们需要先搭建MindSpore环境。首先选择想要安装的MindSpore版本，并执行命令：set MS_VERSION=版本号；接着根据系统环境信息在官网获取安装命令，如下图所示是作者选择的安装命令；最后执行命令：python -c "import mindspore;mindspore.set_context(device_target='CPU');mindspore.run_check()"以验证是否成功安装，如果看到命令行中输出MindSpore version: 版本号，则说明安装成功，如下图所示。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo39.png" width="60%">

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo40.png" width="60%">

***
<a name="学习路径"></a>
## 学习路径

对深度学习有了一定的了解之后，我们会知道深度学习离不开神经网络，可以从多层感知器(Multilayer Perceptron，简称 MLP)神经网络入手。 MLP是最基本的神经网络模型之一，它的结构比较简单，涉及的很多算法是我们学习更复杂的模型的基础，易于理解和实现，同时又有很好的可扩展性和通用性，可以应用于分类、回归等多种任务。学习 MLP 之后，你可以进一步学习卷积神经网络(Convolutional Neural Networks，简称 CNN)和循环神经网络(Recurrent Neural Networks，简称 RNN)等等，它们分别用于计算机视觉和自然语言处理等特定领域的问题。最后我们了解强化学习，它适用于序贯决策问题(涉及一系列有序的决策问题)。学习完各个算法的原理之后，我们可以进行简单的代码实现。

<a name="基本介绍"></a>
## 基本介绍

<a name="何为深度学习"></a>
### 何为深度学习

从定义上说，深度学习是一种机器学习方法，它通过模拟人类大脑的工作原理来处理和分析大量数据，其核心是神经网络，它由多个层次的神经元组成，每一层神经元都负责处理不同的特征。深度学习通过多层次的神经网络来提取数据的特征，并利用这些特征进行分类、预测和其他任务。

下图展示了人工智能、机器学习、深度学习之间的关系：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo1.png" width="60%">


深度学习可被定义为以下四个基本网络框架中具有大量参数和层数的神经网络：
- 无监督预训练网络
- 卷积神经网络
- 循环神经网络
- 递归神经网络


可见深度学习与神经网络是密不可分的，因此我们接下来要介绍的就是神经网络。
***
<a name="神经网络"></a>
### 神经网络
神经网络的学习方法和算法是深度学习的重要组成部分，神经网络在很多领域都有广泛的应用，例如计算机视觉、自然语言处理、语音识别等。

**神经元**

神经网络由大量的神经元相互连接而成。每个神经元接受线性组合的输入后，最开始只是简单的线性加权，后来给每个神经元加上了非线性的激活函数，从而进行非线性变换后输出。每两个神经元之间的连接代表加权值，称之为权重(weight)。不同的权重和激活函数，则会导致神经网络不同的输出。

把单个神经元组织在一起，便形成了神经网络。神经网络的每一可能由单个或多个神经元组成，每一层的输出将会成为下一层的输入数据。下面是一些常见的神经网络类型：

1.前馈神经网络(Feedforward Neural Network)：前馈神经网络是最基本的神经网络类型，也是深度学习中最常见的神经网络类型。它由若干个神经元按照一定的层次结构组成，每个神经元接收上一层的输出，产生本层的输出，从而实现信息的传递和处理。

2.卷积神经网络(Convolutional Neural Network)：卷积神经网络是一种专门用于图像处理和计算机视觉任务的神经网络类型。它通过卷积和池化等操作，可以提取图像中的特征，从而实现图像分类、目标检测、图像分割等任务。

3.循环神经网络(Recurrent Neural Network)：循环神经网络是一种能够处理序列数据的神经网络类型。它通过记忆单元和门控机制等方式，可以处理任意长度的序列数据，从而实现自然语言处理、语音识别等任务。

4.自编码器(Autoencoder)：自编码器是一种无监督学习的神经网络类型，它的目标是将输入数据进行压缩和解压缩，从而实现特征提取和降维等任务。

5.深度置信网络(Deep Belief Network)：深度置信网络是一种由多个受限玻尔兹曼机组成的神经网络类型。它可以通过逐层贪心预训练和微调等方式，实现高效的特征学习和分类任务。

除了以上列举的几种神经网络类型，还有众多其他的神经网络类型，如反向传播神经网络、Hopfield网络、Boltzmann机等。不同的神经网络类型适用于不同的任务和数据类型，需要根据具体的问题选择合适的神经网络类型。

**神经网络组成**

人工神经网络（Artificial Neural Networks，简写为ANNs）是一种模仿动物神经网络行为特征，进行分布式并行信息处理的算法数学模型。这种网络依靠系统的复杂程度，通过调整内部大量节点之间相互连接的关系，从而达到处理信息的目的，并具有自学习和自适应的能力。神经网络类型众多，其中最为重要的是多层感知机。因此为了详细地描述神经网络，我们先从最简单的神经网络说起。

人工神经网络由神经元模型构成，这种由许多神经元组成的信息处理网络具有并行分布结构。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo2.png" width="70%">

```
其中圆形节点表示一个神经元，方形节点表示一组神经元。
```

<a name="项目地址"></a>
### 项目地址
- 模型仓库：[MindSpore/hepucuncao/DeepLearning](https://xihe.mindspore.cn/projects/hepucuncao/DeepLearning)

<a name="项目结构"></a>
### 项目结构

项目的目录分为两个部分：学习笔记readme文档，以及其中一些模型的简单实现代码放在train文件夹下。

```python
 ├── train    # 相关代码目录
 │  ├── MLP.py    # MLP的一个简单实现
 │  ├── CNN.py    # CNN的一个简单实现
 │  └── RNN.py    # RNN的一个简单实现
 │  └── GD.py     # 梯度下降法的一个简单案例
 └── README.md
```



***
<a name="常用模型及算法"></a>
## 常用模型及算法

<a name="多层感知机(MLP)"></a>
### 多层感知机(MLP)

**感知机**

多层感知机(MLP，Multilayer Perceptron)是最简单的深度学习模型，属于前馈神经网络的一种，由多个全连接的神经网络层组成，适用于解决分类和回归问题，也叫人工神经网络(ANN,Artificial Neural Network)。简单的感知机如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo4.png" width="30%">

其中x1,x2,...,xn为感知机的输入，ω的计算与输入是无关的，相当于一个偏置。

除了输入输出层，它中间可以有多个隐层，最简单的MLP只含一个隐层，即三层的结构，且层与层之间是全连接的，如下图：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo3.png" width="40%">

**多层感知机**

多层感知机由感知机推广而来，最主要的特点是有多个神经元层，因此也叫深度神经网络。相比于单独的感知机，多层感知机的第i层的每个神经元和第i−1、i+1层的每个神经元都有连接。且输出层可以不止有1个神经元。隐藏层可以只有1层，也可以有多层。输出层为多个神经元的神经网络例如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo5.png" width="40%">

隐藏层与输入层是全连接的，假设输入层用向量X表示，则隐藏层的输出就是 f (ω1*X+b1)，ω1是权重(也叫连接系数)，权重越高说明这个特征越重要，b1是偏置，函数f可以是常用的sigmoid函数(也称Logistic函数)或者tanh函数:

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo6.png" width="20%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo7.png" width="20%">

隐藏层到输出层可以看成是一个多类别的逻辑回归，也即softmax回归，所以输出层的输出就是softmax(ω2*X1+b2)，可以将输出结果正规化处理，这同样也是通过矩阵运算进行的，其中X1表示隐藏层的输出f(ω1*X+b1)。

**输出的正规化**

我们可以利用以下公式来将输出结果正规化，使得所有元素的和为1，而每个元素的值代表了概率值，此时的神经网络将变成如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo8.png" width="40%">

**激活层**

通过上述两个线性方程的计算，我们可以得到线性输出，接着要对神经网络注入灵魂：激活层。

激活层是神经网络中的一种层，其作用是在输入信号和输出信号之间添加一个非线性的转换函数，增加神经网络模型的非线性，使得网络可以更好地学习和表示复杂的非线性关系。激活层的意义在于增加模型的非线性表达能力，使得神经网络可以更好地处理复杂的输入数据，例如图像、文本和语音等。

激活层常用的激活函数三种，分别是阶跃函数、Sigmoid和ReLU，如下图：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo9.png" width="60%">

- 阶跃函数：当输入小于等于0时，输出0；当输入大于0时，输出1。
- Sigmoid：当输入趋近于正无穷/负无穷时，输出无限接近于1/0，这个函数可以把一个实数压缩到0-1之间。
- ReLU：当输入小于0时，输出0；当输入大于0时，输出等于输入。
```
    注意：每个隐藏层计算（矩阵线性运算）之后，都需要加一层激活层，要不然该层线性计算是没有意义的。此时的神经网络变成了如下图所示的形式：
```
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo10.png" width="40%">


```
一些选择激活函数的经验法则：
如果输出是0、1值(二分类问题)，则输出层选择sigmoid函数，然后其它的所有单元都选择Relu函数。这是很多激活函数的默认选择，如果在隐藏层上不确定使用哪个激活函数，那么通常会使用Relu激活函数。有时，也会使用tanh激活函数。
```


**交叉熵损失**

通过Softmax层的正规化处理后，我们可以得到I、II、III和IV这四个类别分别对应的概率，但这只是神经网络计算得到的而非真实情况。因此，我们需要将SOftmax输出结果的好坏程度做一个“量化”，即将结果取对数的负数，这时概率越接近100%，该计算结果就越接近于0，说明结果越准确，我们把这个输出称为“交叉熵损失”。而我们训练神经网络的目的，就是尽可能减少“交叉熵损失”。

**反向传播**

神经网络的计算主要有两种：前向传播（foward propagation, FP）作用于每一层的输入，通过逐层计算得到输出结果；反向传播（backward propagation, BP）作用于网络的输出，通过计算梯度由深到浅更新网络参数。

上述过程就是神经网络的正向传播过程，用一句话来总结就是：神经网络的传播是形如Y=ω*X+b的线性矩阵运算，在隐藏层中加入激活层来给矩阵运算加入非线性，输出层的结果经过Softmax层处理为概率值，并通过交叉熵损失来量化当前神经网络的优劣。

接着我们要进行的是反向传播，简而言之，反向传播是一个参数优化的过程，优化对象就是神经网络中的非确定参数ω和b。神经网络可以自动优化，通过反复迭代将输出的概率提高、交叉熵损失值下降，直到得到理想的参数值。

那么，在反向传播过程中，很重要的一点就是：参数如何更新?或者说应该朝着什么方向更新?显然，参数应该是朝着目标损失函数下降最快的方向更新，即朝着梯度方向更新！

在深度学习中，有三种最基本的梯度下降算法：**SGD、BGD、MBGD**，他们各有优劣。根据不同的数据量和参数量，可以选择一种具体的实现形式，在训练神经网络是优化算法大体可以分为两类：1）调整学习率，使得优化更稳定；2）梯度估计修正，优化训练速度。
- 随机梯度下降法 (Stochastic Gradient Descent,SGD)：每次迭代(更新参数)只使用单个训练样本
- 批量梯度下降法 (Batch Gradient Descent,BGD)：每次迭代更新中使用所有的训练样本
- 小批量梯度下降法 (Mini-Batch Gradient Descent,MBGD)：折中了 BGD 和 SGD 的方法，每次迭代使用 batch_size 个训练样本进行计算(一般的mini-batch大小为64~256)

接下来我们简单介绍一下在深度学习中应用十分广泛的梯度下降算法，它的主要目的是通过迭代找到目标函数的最小值，或者收敛到最小值。

**梯度下降法**

梯度是微积分中一个重要的概念:
- 在单变量的函数中，梯度其实就是函数的微分，代表着函数在某个给定点的切线的斜率
- 在多变量函数中，梯度是一个向量，向量有方向，梯度的方向就指出了函数在给定点的上升最快的方向(梯度的反方向就是函数在给定点下降最快的方向，这正是我们所需要的)

数学公式如下：
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo38.png" width="30%">
```
此公式的意义是：J是关于Θ的一个函数，设当前所处的位置为Θ0点，要从这个点走到J的最小值点，也就是“山底”，首先先确定前进的方向，即梯度的反向，然后走一段距离的步长α(称为学习率或者步长)，走完这个段步长，就到达了Θ1这个点。
注意：梯度前加一个负号，就意味着朝着梯度相反的方向前进，如果是梯度上升算法，当然就不需要添加负号了。
```

下面是用python实现的一个简单的梯度下降算法拟合直线的案例：

代码地址：[MindSpore/hepucuncao/DeepLearning/train/GD.py](https://xihe.mindspore.cn/projects/hepucuncao/DeepLearning/blob/train/GD.py)

至此，多层感知器(MLP)算法的内容我们就基本介绍完了，它们也是接下来要讲的算法的基础。

下面是用python实现的多层感知机(MLP)算法的一个简单实现：

代码地址：[MindSpore/hepucuncao/DeepLearning/train/MLP.py](https://xihe.mindspore.cn/projects/hepucuncao/DeepLearning/blob/train/MLP.py)

***
<a name="卷积神经网络(CNN)"></a>
### 卷积神经网络(CNN)

卷积神经网络(Convolutional Neural Networks, CNN)是一类包含卷积计算且具有深度结构的前馈神经网络(Feedforward Neural Networks)，是深度学习的代表算法之一。目前CNN 已经得到了广泛的应用，CNN最擅长的就是图片的处理，它受到人类视觉神经系统的启发。

**卷积神经网络的层级结构**

最左边是数据输入层：
对数据做一些处理，CNN只对训练集去均值(把输入数据各个维度都中心化为0，避免数据过多偏差，影响训练效果)。

中间是：
- CONV：卷积计算层，即线性乘积求和，主要作用是保留图片的特征。**(核心)**
- RELU：激励层，ReLU是激活函数的一种。
- POOL：池化层，即取区域平均或最大，主要作用是把数据降维，可以有效的避免过拟合。

最右边是
- FC：全连接层，主要作用是根据不同任务输出我们想要的结果。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo11.png" width="60%">

**卷积计算层**

通俗来说，将未知图案的局部和标准图案的局部一块一块地进行比对的计算过程，便是卷积操作，这个拿来比对的“小块”称之为Features(特征)。卷积计算结果为1表示匹配，否则不匹配。这个过程涉及到了一些数学操作，就是我们常说的“卷积”，因此我们先来了解一下什么是卷积。

**什么是卷积**

对图像(不同的数据窗口数据)和滤波矩阵(一组固定的权重：因为每个神经元的多个权重固定，所以又可以看做一个恒定的滤波器filter做内积(逐个元素相乘再求和)的操作就是所谓的『卷积』操作，这也是卷积神经网络的名字来源。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo12.png" width="50%">

上图中红框框起来的部分可以理解为一个滤波器，即带着一组固定权重的神经元，不同的滤波器filter会得到不同的输出数据，多个滤波器叠加便成了卷积层，用来提取图像的不同特征。在CNN中，滤波器filter对局部输入数据进行卷积计算，每计算完一个数据窗口内的局部数据后，数据窗口不断平移滑动，直到计算完所有数据。这个过程中有以下参数： 

　　a. 深度(depth)：神经元个数，决定输出的depth厚度。同时代表滤波器个数。

　　b. 步长(stride)：决定滑动多少步可以到边缘。

　　c. 填充值(zero-padding)：在外围边缘补充若干圈0，方便从初始位置以步长为单位可以刚好滑倒末尾位置，通俗地讲就是为了总长能被步长整除 (一定程度上也可以弥补边界特征利用不充分的问题)。有三种填充方式：全零填充padding='same'、不填充padding='valid'、自定义填充padding=[[0,0],[上,下],[左,右],[0,0]]。

数据窗口变化的过程中，每次滤波器都是针对某一局部的数据窗口进行卷积，这就是所谓的CNN中的**局部感知机制**。与此同时，中间滤波器Filter的权重是固定不变的，这就是CNN中的**参数(权重)共享机制**，这是卷积层最主要两个特征。
```
1.通过卷积操作实现局部连接，这个局部区域的大小就是滤波器filter，避免了全连接中参数过多造成无法计算的情况。
2.再通过参数共享来缩减实际参数的数量，为实现多层网络提供了可能。
```

下面我们来分析一下具体的计算过程，如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo13.png" width="50%">

类似ω*X+b，其中ω对应滤波器Filter w0，X对应不同的数据窗口，b对应Bios b0，简单来说就是滤波器Filter w0与一个个数据窗口相乘再求和后，最后加上Bias b0得到输出结果。然后滤波器Filter w0固定不变，数据窗口向右移动步长stride个单位，继续做内积计算，重复这个过程。
```
    注意：这里并不同于矩阵中的乘法，而是对应位置相乘后再求和。上图展示的是**多通道输入，单卷积核**的卷积操作(但是一个卷积核可以有多个通道,默认情况下，卷积核的通道数等于输入图片的通道数)。除此之外，还有**单通道输入，单卷积核**和**多通道输入，多卷积核**两种卷积操作，其中多通道输入、多卷积核是深度神经网络中间最常见的形式。
    总结：输出的通道数=卷积核的个数   卷积核的通道数=输入的通道数   偏置数=卷积核数
```

**ReLU激励层**

在MLP算法中我们介绍了sigmoid函数，但在实际梯度下降的过程中，sigmoid函数容易饱和，造成梯度传递终止。因此我们可以使用另一个激活函数：ReLU，它的优点是收敛快且求梯度比较简单，其图形表示如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo14.png" width="30%">

**池化pool层**

在卷积层中，可以通过调节步长参数实现特征图的高宽成倍缩小，从而降低了网络的参数量。实际上，除了通过设置步长，还有一种专门的网络层可以实现尺寸缩减功能，它就是我们要介绍的池化层(Pooling layer)。通常我们用到两种池化进行下采样：(1)**最大池化(Max Pooling)**，从局部相关元素集中选取最大的一个元素值;(2)**平均池化(Average Pooling)**，从局部相关元素集中计算平均值并返回。如下图展示的就是最大池化的操作：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo15.png" width="50%">

**全连接层FC**

之所以叫做全连接，是因为每个神经元与前后相邻层的每一个神经元都有连接关系。如下图所示，这是一个简单的两层全连接网络，输入特征，输出的是预测的结果。而实际应用中一般不会将原始图片直接喂入全连接网络，会先对原始图像进行卷积特征提取，把提取到的特征喂给全连接网络，再让全连接网络计算出分类评估值。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo16.png" width="40%">

全连接层的参数量是可以直接计算的，计算公式为：**参数个数=∑(前层*后层+后层)**

和多层神经网络一样，卷积神经网络中的参数训练也是使用误差反向传播算法。

至此，我们介绍完了深度学习中的一个经典算法--卷积神经网络模型(CNN)，接下来我们要介绍的是另一种也非常经典的算法--循环神经网络模型(RNN)。

下面是用python实现的卷积神经网络(CNN)算法的一个简单实现：

代码地址：[MindSpore/hepucuncao/DeepLearning/train/CNN.py](https://xihe.mindspore.cn/projects/hepucuncao/DeepLearning/blob/train/CNN.py)

***
<a name="循环神经网络(RNN)"></a>
### 循环神经网络(RNN)

**什么是RNNs**

在传统的神经网络模型中，是从输入层到隐含层再到输出层，层与层之间是全连接的，每层之间的节点是无连接的，但是这种普通的神经网络对于很多问题却无能无力。RNNs之所以称为循环神经网络，是因为一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。下图是一个典型的RNNs：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo17.png" width="50%">

```
    从图中可以看出，有一条单向流动的信息流是从输入单元到达隐藏单元的，与此同时另一条单向流动的信息流从隐藏单元到达输出单元。在某些情况下，RNNs也会引导信息从输出单元返回隐藏单元，称为“Back Projections”，并且隐藏层的输入还包括上一隐藏层的状态，即隐藏层内的节点可以自连也可以互连。
```

上图中每一步都会有输出，但是并不是每一步都要有输出，当然也不是每步都需要输入。RNNs的关键之处在于**隐藏层**，隐藏层能够捕捉序列的信息。

下面是一些常见的RNNs模型：

**Simple RNNs(SRNs)**

SRNs是RNNs的一种特例，它是一个三层网络，并且在隐藏层增加了上下文单元。上下文单元节点与隐藏层中的节点的连接是固定的，并且权值也是固定的。在每一步中，使用标准的前向反馈进行传播，然后使用学习算法进行学习。上下文每一个节点保存其连接的隐藏层节点的上一步的输出，即保存上文，并作用于当前步对应的隐藏层节点的状态，即隐藏层的输入由输入层的输出与上一步的自己的状态所决定的，因此SRNs能够解决标准的多层感知机(MLP)无法解决的对序列数据进行预测的任务。SRNs网络结构如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo18.png" width="30%">

**Bidirectional RNNs**

Bidirectional RNNs(双向网络)的改进之处便是，假设当前的输出(第步的输出)不仅仅与前面的序列有关，并且还与后面的序列有关。例如：预测一个语句中缺失的词语那么就需要根据上下文来进行预测。Bidirectional RNNs是一个相对较简单的RNNs，是由两个RNNs上下叠加在一起组成的,输出由这两个RNNs的隐藏层的状态决定的。如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo19.png" width="40%">

**Deep(Bidirectional)RNNs**

Deep(Bidirectional)RNNs与Bidirectional RNNs相似，只是对于每一步的输入有多层网络，使该网络有更强大的表达与学习能力，但是复杂性也提高了，同时需要更多的训练数据。Deep(Bidirectional)RNNs的结构如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo20.png" width="30%">

**Echo State Networks(ESNs)**

ESNs(回声状态网络)也是一种RNNs，但是它与传统的RNNs相差很大。ESNs具有以下三个特点：
- 它的核心结构时一个随机生成、且保持不变的储备池(Reservoir)，储备池是大规模的、随机生成的、稀疏连接(SD通常保持1%～5%，SD表示储备池中互相连接的神经元占总的神经元个数N的比例)的循环结构；
- 其储备池到输出层的权值矩阵是唯一需要调整的部分；
- 简单的线性回归就可完成网络的训练。


从结构上讲，ESNs是一种特殊类型的循环神经网络，其基本思想是：使用大规模随机连接的循环网络取代经典神经网络中的中间层，从而简化网络的训练过程。因此ESNs的关键是中间的储备池。网络中的参数包括：为储备池中节点的连接权值矩阵，为输入层到储备池之间的连接权值矩阵，表明储备池中的神经元之间是连接的，为输出层到储备池之间的反馈连接权值矩阵，表明储备池会有输出层来的反馈，为输入层、储备池、输出层到输出层的连接权值矩阵，表明输出层不仅与储备池连接，还与输入层和自己连接。表示输出层的偏置项。

以下是ESNs储备池的四个参数：
- 储备池内部连接权谱半径SR(只有SR <1时，ESNs才能具有回声状态属性)
- 储备池规模N(即储备池中神经元的个数)
- 储备池输入单元尺度IS(IS为储备池的输入信号连接到储备池内部神经元之前需要相乘的一个尺度因子)
- 储备池稀疏程度SD(即为储备池中互相连接的神经元个数占储备池神经元总个数的比例)

对于IS，如果需要处理的任务的非线性越强，那么输入单元尺度就越大。该原则的本质就是通过输入单元尺度IS，将输入变换到神经元激活函数相应的范围(神经元激活函数的不同输入范围，其非线性程度不同)。ESNs的结构如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo21.png" width="40%">

**Gated Recurrent Unit Recurrent Neural Networks(GRUs)**

GRUs主要是从以下两个方面进行改进:一是，序列中不同的位置处的输入对当前的隐藏层的状态的影响不同，越前面的影响越小，即每个前面状态对当前的影响进行了距离加权，距离越远，权值越小；二是，在产生误差error时，误差可能是由某一个或者几个输入而引发的，所以应当仅仅对对应的输入weight进行更新。GRUs的结构如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo22.png" width="40%">

```
GRUs首先根据当前输入以及前一个隐藏层的状态hidden state计算出update gate和reset gate，再根据reset gate、当前输入以及前一个hidden state计算新的记忆单元内容(new memory content)。当reset gate为1的时候，new memory content忽略之前的所有memory content，最终的memory是之前的hidden state与new memory content的结合。
```

**LSTM Netwoorks(LSTMs)**

LSTMs与GRUs类似，与一般的RNNs结构本质上并没有什么不同，只是使用了不同的函数来计算隐藏层的状态。在LSTMs中，i结构被称为cells，可以把cells看作是黑盒用以保存当前输入之前保存的状态，这些cells决定哪些cell抑制哪些cell兴奋，它们结合前面的状态、当前的记忆以及当前的输入。已经证明，该网络结构在对长序列依赖问题中非常有效，LSTMs的网络结构如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo23.png" width="50%">

**Clockwork RNNs(CW-RNNs)**

CW-RNNs是一种使用时钟频率来驱动的RNNs。它将隐藏层分为几个块(组，Group/Module)，每一组按照自己规定的时钟频率对输入进行处理。并且为了降低标准的RNNs的复杂性，CW-RNNs减少了参数的数目，提高了网络性能，加速了网络的训练。CW-RNNs通过不同的隐藏层模块工作在不同的时钟频率下来解决长时间依赖问题。将时钟时间进行离散化，然后在不同的时间点，不同的隐藏层组在工作。因此，所有的隐藏层组在每一步不会都同时工作，这样便会加快网络的训练。并且，时钟周期小的组的神经元的不会连接到时钟周期大的组的神经元，只会周期大的连接到周期小的(可以认为组与组之间的连接是有向的就好了，代表信息的传递是有向的)，周期大的速度慢，周期小的速度快，那么便是速度慢的连速度快的，反之则不成立。。

CW-RNNs与SRNs网络结构类似，也包括输入层(Input)、隐藏层(Hidden)、输出层(Output)，它们之间也有向前连接，即输入层到隐藏层的连接，隐藏层到输出层的连接。但是与SRN不同的是，隐藏层中的神经元会被划分为若干个组，每一组中的神经元个数相同，并为每一个组分配一个时钟周期，每一个组中的所有神经元都是全连接。CW-RNNs的网络结构如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo24.png" width="55%">

```
如上图所示，将这些组按照时钟周期递增从左到右进行排序，那么连接便是从右到左。例如：隐藏层共有256个节点，分为四组，周期分别是[1,2,4,8]，那么每个隐藏层组256/4=64个节点，第一组隐藏层与隐藏层的连接矩阵为64*64的矩阵，第二层的矩阵则为64*128矩阵，第三组为64*(3*64)=64*192矩阵，第四组为64*(4*64)=64*256矩阵。这就解释了上述为什么说，速度慢的组连到速度快的组，反之则不成立。
```

下面是用python实现的循环神经网络(RNN)算法的一个简单实现：

代码地址：[MindSpore/hepucuncao/DeepLearning/train/RNN.py](https://xihe.mindspore.cn/projects/hepucuncao/DeepLearning/blob/train/RNN.py)

RNN在处理长期依赖(时间序列上距离较远的节点)时会遇到巨大困难，因为计算距离较远的节点之间的联系时会涉及雅可比矩阵的多次相乘，会造成梯度消失或者梯度膨胀的现象。其中最成功、应用最广泛的就是门限RNN(Gated RNN)，而LSTM就是门限RNN中最著名的一种,接下来我们介绍LSTM长短期记忆神经网络。

***

<a name="LSTM长短期记忆神经网络"></a>
### LSTM长短期记忆神经网络

**RNN和LSTM的区别**

所有 RNN 都具有一种重复神经网络模块的链式的形式。在标准的 RNN 中，这个重复的模块只有一个非常简单的结构，例如一个 tanh 层，如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo25.png" width="60%">

LSTM 同样是这样的结构，但是重复的模块拥有一个不同的结构。不同于单一神经网络层，这里是有四个，以一种非常特殊的方式进行交互。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo26.png" width="60%">

**LSTM核心**

LSTM 有统称为“门”的结构来去除或增加信息到细胞状态的能力。门是一种让信息选择式通过的方法，它包含一个sigmoid神经网络层和一个pointwise乘法操作。示意图如下：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo27.png" width="60%">

LSTM 拥有三个门，分别是忘记层门，输入层门和输出层门，用来保护和控制细胞状态。

**忘记层门**

作用对象：细胞状态

作用：将细胞状态中的信息选择性遗忘

操作步骤：该门会读取ht−1​​和xt​​，输出一个在 0 到 1 之间的数值给每个在细胞状态Ct−1​​中的数字。1表示“完全保留”，0表示“完全舍弃”。示意图如下：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo28.png" width="60%">

**输入层门**

作用对象：细胞状态

作用：将新的信息选择性的记录到细胞状态中

操作步骤：

步骤一，sigmoid层称“输入门层”决定什么值我们将要更新。

步骤二，tanh 层创建一个新的候选值向量Ct​​加入到状态中。其示意图如下：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo29.png" width="60%">

步骤三：将Ct−1​​更新为Ct​​。将旧状态与ft​​相乘，丢弃掉我们确定需要丢弃的信息。接着加上it​∗Ct​​得到新的候选值，根据我们决定更新每个状态的程度进行变化。其示意图如下：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo30.png" width="60%">

**输出层门**

作用对象：隐层ht​​

作用：确定输出什么值

操作步骤：

步骤一：通过sigmoid层来确定细胞状态的哪个部分将输出。

步骤二：把细胞状态通过tanh进行处理，并将它和sigmoid门的输出相乘，最终仅输出我们确定输出的那部分。其示意图如下所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo31.png" width="60%">

***

<a name="强化学习(RL)"></a>
### 强化学习(RL)

强化学习Reinforcement Learning (RL)属于机器学习的一种，不同于监督学习和无监督学习，它是通过智能体与环境的不断交互进而获得奖励，从而不断优化自身动作策略，以期待最大化其长期收益。强化学习适用于序贯决策问题(涉及一系列有序的决策问题)。

**强化学习的定义**

智能体与环境的不断交互(即在给定状态采取动作)，进而获得奖励，此时环境从一个状态转移到下一个状态。智能体通过不断优化自身动作策略，以期待最大化其长期回报或收益(奖励之和)。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo32.png" width="50%">

**深度强化学习**

Deep Learning(DL) + Reinforcement Learning(RL) = Deep Reinforcement Learning(DRL)

深度学习DL有很强的抽象和表示能力，特别适合建模RL中的值函数，二者结合，极大地拓展了RL的应用范围。深度强化学习的算法比较多，常见的有：DQN，DDPG，PPO，TRPO，A3C，SAC等。

**Deep Q-Networks(DQN)算法**

DQN，即深度Q网络（Deep Q-network），是指基于深度学习的Q-Learing算法。

**1)DQN训练过程**

神经网络的的输入是状态s，输出是对所有动作a的打分

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo33.png" width="60%">

最原始的DQN算法具体过程如下：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo34.png" width="50%">

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo35.png" width="50%">

从上述过程我们可以知道，1、用完一个transition就丢弃，会造成对经验的浪费，且按顺序使用transition时，前一个transition和后一个transition相关性很强，这种相关性对学习Q网络是有害的。因此，出现了**经验回放**，它可以克服上述两个缺点。

**2)经验回放**

经验回放会构建一个回放缓冲区(replay buffer)，存储n条transition，称为经验。当某一个策略π与环境交互，收集很多条transition放入回放缓冲区，回放缓冲区中的经验transition可能来自不同的策略，且回放缓冲区只有在它装满的时候才会吧旧的数据丢掉。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo36.png" width="50%">

每次随机抽出一个batch大小的transition数据训练网络，算出多个随机梯度，用梯度的平均值来更新Q网络参数ω。

**3)目标网络**
```
我们为什么要使用目标网络?
我们在训练网络的时候，动作价值估计和权重w有关。当权重变化时，动作价值的估计也会发生变化。在学习的过程中，动作价值试图追逐一个变化的回报，容易出现不稳定的情况。
```

我们使用另一个网络，称为目标网络:Q(s,a;w1)，网络结构和原来的网络Q(s,a;w)一样，只是参数不同w1 ≠ w，原来的网络称为**评估网络**。两个网络的作用不一样：评估网络Q(s,a;w)负责控制智能体，收集经验，而目标网络Q(s,a;w1)用于计算TD target。

在更新过程中，我们只更新评估网络Q(s,a;w)的权重w，目标网络Q(s,a;w1)的权重w1保持不变。在更新一定次数后，再将更新过的评估网络的权重复制给目标网络，进行下一批更新，这样目标网络也能得到更新。由于在目标网络没有变化的一段时间内回报的目标值是相对固定的，因此目标网络的引入可以增加学习的稳定性。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/photo/photo37.png" width="50%">

**Deep Deterministic Policy Gradient(DDPG)算法**

DDPG算法可以看作Deterministic Policy Gradient(DPG)算法和深度神经网络的结合，是对上述深度Q网络(DQN)在连续动作空间的扩展，它在许多连续控制问题上取得了非常不错的效果。具体而言，DDPG算法主要包括以下三个关键技术：

**1)经验回放**

智能体将得到的经验数据(s,a,r,s′,done)放入Replay Buffer中，更新网络参数时按照批量采样,即**存储**和**回放**:
- 存储:将经验以(s,a,r,s′,done)形式存储在经验池中(集中式回放or分布式回放)
- 回放:按照某种规则从经验池中采样一条或多条经验数据(均匀回放or优先回放)

**2)目标网络**

在Actor网络和Critic网络外再使用一套用于估计目标的Target Actor网络和Target Critic网络。

在更新目标网络时，为了避免参数更新过快，采用软更新方式，也可以称为指数平均移动(Exponential Moving Average, EMA)，即引入一个学习率(或者称为动量)τ，将旧的目标网络参数和新的对应网络参数做加权平均，然后赋值给目标网络。因此，目标网络的输出会更加稳定，利用目标网络来计算目标值自然也会更加稳定，从而进一步保证Critic网络的学习过程更加平稳。

上述作用是其中之一，引入目标网络还有另一个作用就是:避免自举(Bootstrapping)问题。自举是指用后继的估算值，来更新现在状态的估算值，它会使网络出现过估计的问题。如果过估计是均匀的，对于最终的决策不会造成影响；但是如果不均匀，对于最终的决策会产生很大影响。

**3)噪声探索**

探索对于智能体来说是至关重要的，确定性策略输出的动作为确定性动作，缺乏对环境的探索。因此在训练阶段，要人为地给Actor网络输出的动作加入噪声，从而让智能体具备一定的探索能力。

```
注意:噪声只会加在训练阶段Actor网络输出的动作上，推理阶段不要加上噪声，以及在更新网络参数时也不要加上噪声，因为我们只需要在训练阶段让智能体具备探索能力，推理时是不需要的该能力的。
```



