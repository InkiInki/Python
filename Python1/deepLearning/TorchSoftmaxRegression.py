'''
@(#)SoftMax.py
The class of SoftMax.
Author: inki
Email: inki.yinji@qq.com
Created on May 08, 2020
Last Modified on May 08, 2020
'''
import warnings
warnings.filterwarnings('ignore')
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from ImageMnist import *
from collections import OrderedDict
        
def train_softmax(net, train_iter, test_iter, loss, num_epochs, 
                      batch_size, params=None, lr=None, optimizer=None):
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n = 0., 0., 0
            for x, y in train_iter:
                y_hat = net(x)
                l = loss(y_hat, y).sum()
                
                if optimizer is not None:
                    optimizer.zero_grad()
                elif params is not None and params[0].grad is not None:
                    for param in params:
                        param.grad.data.zero_()
                        
                l.backward()
                if optimizer is None:
                    sgd(params, lr, batch_size)
                else:
                    optimizer.step()
                    
                train_l_sum += l.item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                n += y.shape[0]
            test_acc = evaluate_accuracy(test_iter, net)
            print("Epoch %d, loss %.4f, train acc %.3f, test acc %.3f" %
                  (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
            
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size
        
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0., 0
    for x, y in data_iter:
        acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
    
class FlattenLayer(nn.Module):
    
    def __init__(self):
        super(FlattenLayer, self).__init__()
        
    def forward(self, x):
        return x.view(x.shape[0], -1)        

if __name__ == '__main__':
    batch_size = 256
    num_inputs=784
    num_outputs=10
    num_epochs = 5
    im = ImageMnist()
    im.__init__()
    train_iter, test_iter = im.data_iter(batch_size)
    net = nn.Sequential(
        OrderedDict([
            ('flatten', FlattenLayer()),
            ('linear', nn.Linear(num_inputs, num_outputs))])
        )
    init.normal(net.linear.weight, mean=0, std=0.01)
    init.constant_(net.linear.bias, val=0)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
    
    
    
    
    
    
    
    
    
    
    
    
    