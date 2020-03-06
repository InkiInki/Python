'''
@(#)SimpleTool.py
The class of test.
Author: inki
Email: inki.yinji@qq.com
Created on May 05, 2020
Last Modified on May 06, 2020
'''
# coding = utf-8

import torch
import warnings
warnings.filterwarnings('ignore')
from data.benchmark import *

__all__ = ['euclidean_metric',
           'hausdorff_average_metric',
           'hausdorff_maximum_metric',
           'hausdorff_minimum_metric',
           'hausdorff_virtual_metric',
           'load_arff\n']

def introduction(__all__=__all__):
    _num_function = 0
    print("The function list:")
    for temp in __all__:
        _num_function = _num_function + 1
        print("%d-st: %s" % (_num_function, temp))

def euclidean_metric(instance1, instance2):
    """euclidean metric"""
    return torch.sqrt(torch.tensor(sum((instance1 - instance2)**2), dtype=torch.float))

def hausdorff_average_metric(instances1, instances2):
    """average hausdorff metric for multi-instance learning"""
    len_instances1 = len(instances1)
    len_instances2 = len(instances2)
    
    temp_sum = 0
    for i in range(len_instances1):
        temp_list = torch.zeros(len_instances2)
        for j in range(len_instances2):
            temp_list[j] = euclidean_metric(instances1[i], instances2[j])
        temp_sum += min(temp_list)
    
    for j in range(len_instances2):
        temp_list = torch.zeros(len_instances1)
        for i in range(len_instances1):
            temp_list[i] = euclidean_metric(instances2[j], instances1[i])
        temp_sum += min(temp_list)
    
    return temp_sum / (len_instances1 + len_instances2)
            
    
def hausdorff_maximum_metric(instances1, instances2):
    """maximum hausdorff metric for multi-instance learning"""
    len_instances1 = len(instances1)
    len_instances2 = len(instances2)
    
    temp_max1 = torch.zeros(len_instances1)
    for i in range(len_instances1):
        temp_min = torch.zeros(len_instances2)
        for j in range(len_instances2):
            temp_min[j] = euclidean_metric(instances1[i], instances2[j])
        temp_max1[i] = min(temp_min)
    
    temp_max2 = torch.zeros(len_instances2)
    for j in range(len_instances2):
        temp_min = torch.zeros(len_instances1)
        for i in range(len_instances1):
            temp_min[i] = euclidean_metric(instances2[j], instances1[i])
        temp_max2[j] = min(temp_min)
    
    return max(max(temp_max1), max(temp_max2))
    
    
    
def hausdorff_minimum_metric(instances1, instances2):
    """minimum hausdorff metric for multi-instance learning"""
    len_instances1 = len(instances1)
    len_instances2 = len(instances2)
    sum_len = len_instances1 + len_instances2
    
    k = 0
    temp_min = torch.zeros(sum_len)
    for i in range(len_instances1):
        for j in range(len_instances2):
            temp_min[k] = euclidean_metric(instances1[i], instances2[j])
            k += 1
    
    return min(temp_min)
    
def hausdorff_virtual_metric(instances1, instances2):
    """virtual hausdorff metric for multi-instance learning"""
    len_instances1 = len(instances1)
    len_instances2 = len(instances2)
    
    sum1 = torch.zeros(len(instances1[0]))
    for i in range(len_instances1):
        sum1 += instances1[i]
    sum1 /= len_instances1
    
    sum2 = torch.zeros(len(instances2[0]))
    for j in range(len_instances2):
        sum2 += instances2[j]
    sum2 /= len_instances2
        
    
    return euclidean_metric(sum1, sum2)

def load_file(file_name):
    """load file, return data"""
    with open(file_name) as fd:
        fd_datas = fd.readlines()
    
    return fd_datas
    
if __name__ == '__main__':
    instance1 = torch.tensor([[0, 0, 1], [0, 0, 0]])
    instance2 = torch.tensor([[1, 0, 0], [1, 0, 0]])
    print(hausdorff_virtual_metric(instance1, instance2))
