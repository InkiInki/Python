'''
@(#)test1.py
The class of test.
Author: inki
Email: inki.yinji@qq.com
Created on May 4, 2020
Last Modified on May 4, 2020
'''
import torch
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from IPython import display
from matplotlib import pyplot as plt

class TorchLinearRegression(nn.Module):
    
    def _init_(self, num_instances=1000, num_attributions=2, true_w=[2, -3.4], 
               true_b=4.2, epsilon=[0, 0.01], batch_size=10, lr=0.03, num_epochs=3, weight=[0, 0.01], bias=0):
        self.num_instances = num_instances
        self.num_attributions = num_attributions
        self.true_w = true_w
        self.true_b = true_b
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.weight = weight
        self.bias = bias
        self.net = nn.Sequential(nn.Linear(test.num_attributions, 1))

    def generate_data_set(self):
        datas= torch.tensor(np.random.normal(0, 1, (self.num_instances, self.num_attributions)), dtype=torch.float)
        labels = self.true_w[0] * datas[:, 0] + self.true_w[1] * datas[:, 1] + self.true_b
        labels += torch.tensor(np.random.normal(self.epsilon[0], self.epsilon[1], size=labels.size()), dtype=torch.float)
        return datas, labels
    
    def initialize(self):
        from torch.nn import init
        
        init.normal_(self.net[0].weight, mean=self.weight[0], std=self.weight[1])
        init.constant_(self.net[0].bias, val=self.bias)
    
if __name__ == "__main__":
    test = TorchLinearRegression()
    test._init_()
    datas, labels = test.generate_data_set()
    data_set = Data.dataset.TensorDataset(datas, labels)
    data_iter = Data.DataLoader(data_set, test.batch_size, shuffle=True)
    loss = nn.MSELoss()
    
    optimizer = optim.SGD(test.net.parameters(), lr=test.lr)
    for epoch in range(1, test.num_epochs + 1):
        for x, y in data_iter:
            output = test.net(x)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print('Epoch %d, loss: %f' %(epoch, l.item()))
    print(test.true_w, test.net[0].weight)
    print(test.true_b, test.net[0].bias)














