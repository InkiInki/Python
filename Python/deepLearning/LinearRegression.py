'''
@(#)test1.py
The class of test.
Author: Yu-Xuan Zhang
Email: inki.yinji@qq.com
Created on February 28, 2020
Last Modified on May 4, 2020

@author: inki
'''
import torch
import numpy as np
import random
from IPython import display
from matplotlib import pyplot as plt
from numpy import dtype

class LinearRegression():
    
    def _init_(self, num_instances=1000, num_attributions=2, true_w=[2, -3.4], 
               true_b=4.2, epsilon=[0, 0.01], batch_size=10, w=[0, 0.01], lr=0.03,
               num_epochs=3):
        self.num_instances = num_instances
        self.num_attributions = num_attributions
        self.true_w = true_w
        self.true_b = true_b
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.w = torch.tensor(np.random.normal(w[0], w[1], (self.num_attributions, 1)), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.lr = lr
        self.num_epochs = num_epochs

    def generate_data_set(self):
        datas= torch.from_numpy(np.random.normal(0, 1, (self.num_instances, self.num_attributions)))
        labels = self.true_w[0] * datas[:, 0] + self.true_w[1] * datas[:, 1] + self.true_b
        labels += torch.from_numpy(np.random.normal(self.epsilon[0], self.epsilon[1], size=labels.size()))
        #self.set_figsize()
        #plt.scatter(datas[:, 1].numpy(), labels.numpy(), s=2)
        #plt.show()
        return datas, labels
     
    def set_figsize(self, figsize=(6, 4)):
        display.set_matplotlib_formats('svg')
        plt.rcParams['figure.figsize'] = figsize
         
    def data_iter(self, datas, labels):
        num_instances = len(datas)
        indices = list(range(num_instances))
        random.shuffle(indices)
        for i in range(0, num_instances, self.batch_size):
            j = torch.LongTensor(indices[i : min(i + self.batch_size, num_instances)])
            yield datas.index_select(0, j), labels.index_select(0, j)
            
    def linreg(self, datas, w, b):
        return torch.mm(datas, w) + b
    
    def squared_loss(self, y_hat, y):
        return (y_hat - y.view(y_hat.size())) ** 2 / 2
    
    def sgd(self, params):
        for param in params:
            param.data -= self.lr * param.grad / self.batch_size

if __name__ == "__main__":
    test = LinearRegression()
    test._init_()
    datas, labels = test.generate_data_set()
    for epoch in range(test.num_epochs):
        for x, y in test.data_iter(datas, labels):
            l = test.squared_loss(test.linreg(x, test.w, test.b), y).sum()
            l.backward()
            test.sgd([test.w, test.b])
            test.w.grad.data.zero_()
            test.b.grad.data.zero_()
        train_l = test.squared_loss(test.linreg(datas, test.w, test.b), labels)
        print("Epoch %d, loss %f" % (epoch + 1, train_l.mean()))
    print("W: ", test.w, ", b: ", test.b)
        
        
        
        
        
        
        