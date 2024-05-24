import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import random
 
#定义激活函数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivate(x):
    return x * (1 - x)  # sigmoid函数的导数
 
def relu(x):
  return np.maximum(0.001*x,x)

def relu_derivate(x):
    if x <0 or x == 0 : 
      return 0.001
    else :              
      return   
 
def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def tanh_derivate(x):
    return 1-(x*x)
 
class MLP(object):
    def __init__(self,lr,epoch,MLPSize):
      self.lr = lr
      self.epoch = epoch
 
      self.input_n = MLPSize[0] + 1
      self.hidden_n = MLPSize[1]
      self.output_n = MLPSize[2]
 
      self.input = self.input_n*[0.1]
      self.hidden = self.hidden_n*[0.1]
      self.output = self.output_n*[0.1]
 
      self.input_weights = np.random.normal(loc=0.0, scale=1.0, size=((self.input_n, self.hidden_n)))
      self.output_weights = np.random.normal(loc=0.0, scale=1.0, size=((self.hidden_n, self.output_n)))
 
    def predict(self, inputs):
        for i in range(self.input_n-1):
            self.input[i] = inputs[i]
 
        for j in range(self.hidden_n):
            total = 0
            for i in range(self.input_n):
                total += self.input[i] * self.input_weights[i][j]
            self.hidden[j] = relu(total)
 
        for k in range(self.output_n):
            total = 0
            for j in range(self.hidden_n):
                total += self.hidden[j] * self.output_weights[j][k] + self.bias_output[k]
            self.output[k] = relu(total)
        return self.output[:]
 
    def back_propagate(self, data, label):
        self.predict(data)
        output_deltas = [0.0] * self.output_n
 
        for o in range(self.output_n):
            error = label[o] - self.output[o]  
            output_deltas[o] = relu_derivate(self.output[o]) * error  
 
        hidden_deltas = [0.0] * self.hidden_n
        for j in range(self.hidden_n):
            error = 0
            for k in range(self.output_n):
                error += output_deltas[k] * self.output_weights[j][k]
            hidden_deltas[j] = relu_derivate(self.hidden[j]) * error
 
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden[h]
                self.output_weights[h][o] += self.lr * change
 
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input[i]
                self.input_weights[i][h] += self.lr * change
        error = 0
        for o in range(len(label)):
            for k in range(self.output_n):
                error += 0.5 * (label[o] - self.output[k]) ** 2
        return error
 
    def train(self, data, labels):
        loss = []
        epoch = []
        for i in range(self.epoch):
            error = 0
            for j in range(len(cases)):
                data = np.squeeze(data [j].tolist())
                label = labels[j]
                error += self.back_propagate(data, label)
            epoch.append(i)
            loss.append(error/len(data))
        return epoch,loss
 
    def fit(self):
        DataPath = r'.\Data\Data.csv'
        file = open(DataPath)
        reader = np.mat(list(csv.reader(file)),dtype=np.float64)
        epoch,loss = self.train(reader[:,:2], reader[:,2:3].tolist())  
        test_x = []
        test_y = []
        test_p = []
 
        yold = 0
 
        for x in np.arange(-15, 25, 0.1):
            for y in np.arange(-10, 10, 0.1):
                yp = self.predict(np.array([x, y]))
                if (yold < 0.5 and yp > 0.5):
                    test_x.append(x)
                    test_y.append(y)
				  yold = yp
         
        plt.figure(1)
        plt.title('MLP')
        plt.xlabel('value')
        plt.ylabel('value')
        plt.plot(test_x, test_y, 'b--')
        plt.plot(np.squeeze(reader[:,0:1].tolist()),
                 np.squeeze(reader[:,1:2].tolist()), 'r*')
        plt.savefig(r'.\Data' + '\\' + 'MLP1.png')
        plt.figure(2)
        plt.title('loss_line')
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.plot(epoch, loss, 'r--')
        plt.savefig(r'.\Data' + '\\' + 'MLP1_loss.png')
        plt.show()
 
if __name__ == '__main__':
    model = MLP(lr = 0.01,epoch=500,MLPSize=[2,5,1])
    model.fit()