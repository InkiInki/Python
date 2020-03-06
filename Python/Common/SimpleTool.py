'''
@(#)SimpleTool.py
The class of test.
Author: Yu-Xuan Zhang
Email: inki.yinji@qq.com
Created on May 05, 2020
Last Modified on May 05, 2020

@author: inki
'''
# coding = utf-8

__all__ = ['load_arff\n']


def introduction():
    _num_function = 0
    print("The function list:")
    for temp in __all__:
        _num_function = _num_function + 1
        print("%d-st: %s" % (_num_function, temp))

def load_arff(file_name):
    """load the arff file, return the data"""
    with open(file_name) as fd:
        fd_datas = fd.readlines()
    
    for data in fd_datas:
        print(data)
    
if __name__ == '__main__':
    load_arff("data/benchmark/MUSK1_2.arff")
        