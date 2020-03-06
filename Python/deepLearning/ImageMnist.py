'''
@(#)test.py
The class of test.
Author: Yu-Xuan Zhang
Email: inki.yinji@qq.com
Created on May 05, 2020
Last Modified on May 05, 2020

@author: inki
'''
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
from IPython import display

class ImageMnist():
    
    def __init__(self):
        self.mnist_train = torchvision.datasets.FashionMNIST(root='~/DataSets/FashionMNIST',
            train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = torchvision.datasets.FashionMNIST(root='~/DataSets/FashionMNIST',
            train=False, download=True, transform=transforms.ToTensor())
        
    def get_text_labels(self, labels):
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]
    
    def show_mnist(self, images, labels):
        display.set_matplotlib_formats('svg')
        _, figs = plt.subplots(1, len(images), figsize=(12, 12))
        for f, img, lbl in zip(figs, images, labels):
            f.imshow(img.view((28, 28)).numpy())
            f.set_title(lbl)
            f.axis('off')
        plt.show()
        
        
if __name__ == "__main__":
    test = ImageMnist()
    test.__init__()
    x, y = [], []
    for i in range(10):
        x.append(test.mnist_train[i][0])
        y.append(test.mnist_train[i][1])
    test.show_mnist(x, test.get_text_labels(y))
    
    