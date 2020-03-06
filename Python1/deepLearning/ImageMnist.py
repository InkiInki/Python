'''
@(#)test.py
The class of test.
Author: inki
Email: inki.yinji@qq.com
Created on May 05, 2020
Last Modified on May 06, 2020
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
        
    def data_iter(self, batch_size=256):
        if sys.platform.startswith('win'):
            num_workers = 0
        else:
            num_workers = 4
        train_iter = torch.utils.data.DataLoader(self.mnist_train, 
            batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_iter = torch.utils.data.DataLoader(self.mnist_test, 
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_iter, test_iter
        
if __name__ == "__main__":
    start = time.time()
    test = ImageMnist()
    test.__init__()
    train_iter, test_iter = test.data_iter()
    for x, y in train_iter:
        continue
    print("%.2f sec" % (time.time() - start))
    
    