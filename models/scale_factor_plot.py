# -*- coding: utf-8 -*-
"""
Created on Tue May  7 22:11:13 2024

@author: 18505
"""
import numpy as np
import os
import matplotlib.pyplot as plt

import json
from tqdm import tqdm
def scale(json_file_root):
    scale_list = []
    for folder in os.listdir(json_file_root): 
        try: 
            json_file = os.path.join(json_file_root, folder, "output.json")
            file = open(json_file)
            data = json.load(file)
            scale_list.append(data['scale'])
        except: 
            continue
    
    return scale_list




if __name__ == "__main__": 
    dataset_path = "D:\\PhDProject_real_data\\"
    dataset_names = ["brunei_2023_bat_test_13_2"]
    indexes = []
    iou_loss_mean = []
    iou_loss_std = []
    f = plt.figure()
    ax = f.add_subplot()
   
    
    for index, dataset_name in tqdm(enumerate(dataset_names)): 
        json_file_root = os.path.join(dataset_path, dataset_name, 'rearrange_pose')
        
        scale_list = scale(json_file_root)
        print(scale_list)
        plt.plot(scale_list)
        plt.xlabel("reconstruction index")
        plt.ylabel("scale factor (1 actual bat size)")
        
        
    
    
    print(len(iou_loss_mean))
    print(len(iou_loss_std))
    print(len(indexes))
    plt.xticks(visible=False)
   
    plt.bar(indexes, iou_loss_mean,edgecolor='grey', color='grey')
    plt.errorbar(indexes, iou_loss_mean, iou_loss_std,ls='none',color='black', elinewidth=1,capthick=1, capsize = 10)
    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    