'''
@(#)test.py
The class of test.
Author: Yu-Xuan Zhang
Email: inki.yinji@qq.com
Created on February 26, 2020
Last Modified on May 05, 2020

@author: inki
'''
import os
from sklearn.datasets import *
from alipy import ToolBox


def read(filename):
    with open(filename) as file_object:
        contends = file_object.read()
        print(contends.strip())
    file_object.close()
    
def content_handle():
    
    print(os.stat(path="test.py", dir_fd=None, follow_symlinks=True))
    
def test():
    len_ind = 10
    index = list(range(len_ind))
    ratio = int(0.7 * len_ind)
    train_ind = index[: ratio]
    test_ind = index[ratio :]
    print("Train index: ", train_ind)
    print("Test index: ", test_ind)

if __name__ == "__main__":
    test()