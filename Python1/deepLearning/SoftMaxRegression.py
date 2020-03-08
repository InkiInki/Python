'''
@(#)SoftMax.py
The class of SoftMax.
Author: inki
Email: inki.yinji@qq.com
Created on May 06, 2020
Last Modified on May 08, 2020
'''
import torch
import torchvision
import numpy as np
import sys
import torch.optim as optim
from ImageMnist import *


class SoftMaxRegression():
    
    def __init__(self, num_inputs=784, num_outputs=10, batch_size=256):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.w = torch.tensor(np.random.normal(0, 0.01, (self.num_inputs, self.num_outputs)), 
                              dtype=torch.float, requires_grad=True)
        self.b = torch.zeros(self.num_outputs, dtype=torch.float, requires_grad=True)
        
    def softmax(self, x):
        x_exp = x.exp()
        partition = x_exp.sum(dim=1, keepdim=True)
        return x_exp / partition
    
    def net(self, x):
        return self.softmax(torch.mm(x.view((-1, self.num_inputs)), self.w) + self.b)
    
    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat.gather(1, y.view(-1, 1)))
    
    def accuracy(self, y_hat, y):
        return (y_hat.argmax(dim=1) == y).float().mean().item()
    
    def evaluate_accuracy(self, data_iter, net):
        acc_sum, n = 0., 0
        for x, y in data_iter:
            acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n
    
    def train_softmax(self, train_iter, test_iter, num_epochs, 
                      params=None, lr=None, optimizer=None):
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n = 0., 0., 0
            for x, y in train_iter:
                y_hat = self.net(x)
                l = self.cross_entropy(y_hat, y).sum()
                
                if optimizer is not None:
                    optimizer.zero_grad()
                elif params is not None and params[0].grad is not None:
                    for param in params:
                        param.grad.data.zero_()
                        
                l.backward()
                if optimizer is None:
                    self.sgd(params, lr)
                else:
                    optimizer.step()
                    
                train_l_sum += l.item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                n += y.shape[0]
            test_acc = self.evaluate_accuracy(test_iter, self.net)
            print("Epoch %d, loss %.4f, train acc %.3f, test acc %.3f" %
                  (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
            
    def sgd(self, params, lr):
        for param in params:
            param.data -= lr * param.grad / self.batch_size
        
if __name__ == '__main__':
    test = SoftMaxRegression()
    im = ImageMnist()
    im.__init__()
    train_iter, test_iter = im.data_iter(test.batch_size)
    x, y = iter(test_iter).next()
    
    true_labels = im.get_text_labels(y.numpy())
    pred_labels = im.get_text_labels(test.net(x).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    im.show_mnist(x[ : 9], titles[ : 9])