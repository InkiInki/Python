'''
@(#)Main.py
The class of test.
Author: inki
Email: inki.yinji@qq.com
Created on May 05, 2020
Last Modified on May 05, 2020
'''
import torch
import sys
from Common import SimpleTool
from sympy.strategies.core import switch

__all__ = []

class Main():
    
    def __init__(self, train_set_ratio=0.9, distance_metrics_choose=0, dc_ratio=0.2):
        """manual setting"""
        self.train_set_ratio = train_set_ratio
        self.distance_metrics_choose = distance_metrics_choose
        self.dc_ratio = dc_ratio
        """automatic setting"""
        self.datas = []
        self.num_bags = 0
        self.num_instances = 0
        self.bags_index = []
        self.bags_size = []
        self.bags_labels = []
        self.num_attributions = 0
        
    def initialize(self, file_name1, file_name2):
        """initialize of multi-instance learning"""
        fd_data1 = SimpleTool.load_file(file_name1)
        fd_data2 = SimpleTool.load_file(file_name2)
        self.num_bags = len(fd_data2)
        self.num_instances = len(fd_data1)
        temp_size = 0
        self.bags_index.append(0)
        for i in range(self.num_bags):
            temp_data = fd_data2[i].split(',')
            temp_size += int(temp_data[0])
            self.bags_index.append(temp_size)
            self.bags_size.append(int(temp_data[0]))
            self.bags_labels.append(float(temp_data[1]))
        for data in fd_data1:
            data = [float(temp_data) for temp_data in data.split(',')]
            self.datas.append(data[: -1])
        self.datas = torch.tensor(self.datas)
        self.num_attributions = len(self.datas[0])
        
    def initialize_distance_metric(self):
        """initialize the distance metric"""
        if self.distance_metrics_choose == 0:
            distance_metrics_choosed = SimpleTool.hausdorff_average_metric
        elif self.distance_metrics_choose == 1:
            distance_metrics_choosed = SimpleTool.hausdorff_maximum_metric
        elif self.distance_metrics_choose == 2:
            distance_metrics_choosed = SimpleTool.hausdorff_minimum_metric
        elif self.distance_metrics_choose == 3:
            distance_metrics_choosed = SimpleTool.hausdorff_virtual_metric
        else:
            print("Fatal error in ../Python/SMDP/Main.initialize_distance_metric: there have not this distance metric")
            sys.exit()
        
        

if __name__ == '__main__':
    main = Main()
    file_name1 = "D:\\program\\Java\\eclipse-workspace\\Python\\data\\benchmark\\MUSK1_1"
    file_name2 = "D:\\program\\Java\\eclipse-workspace\\Python\\data\\benchmark\\MUSK1_2"
    main.__init__()
    main.initialize(file_name1, file_name2)
    print(main.num_attributions)
    