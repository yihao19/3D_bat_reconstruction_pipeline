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
def iou_loss(dataset_root):
    iou_loss = []
    # loop throught the reconstruction and store the name of the folder
    folder_name_list = []
    reconstruction_path = os.path.join(dataset_root, 'reconstruction')
    for reconstruction_name in os.listdir(reconstruction_path): 
        obj_name = reconstruction_name.split('.')[0]
        folder_name = obj_name.split('_')[-1]
        folder_name_list.append(folder_name)
        
    json_file_root = os.path.join(dataset_root, 'rearrange_pose')
    for folder in folder_name_list: 
        try: 
            json_file = os.path.join(json_file_root, folder, "output.json")
            file = open(json_file)
            data = json.load(file)
            iou_loss.append(data['IOU'])
            if(data['IOU'] > 0.3): 
                print(f"{folder}: {data['IOU']}")
        except: 
            continue
    
    return np.mean(iou_loss), np.std(iou_loss), iou_loss




if __name__ == "__main__": 
    dataset_path = "D:\\PhDProject_real_data\\"
    dataset_names = ["brunei_2023_bat_test_13_2", "brunei_2023_bat_test_13_1", 'brunei_2023_bat_test_14_1',
                     "brunei_2023_bat_test_15_1","brunei_2023_bat_test_16_1", "brunei_2023_bat_test_17_1", 
                     "Brunei_2023_bat_14_1", "Brunei_2023_bat_15_1","Brunei_2023_bat_15_2","Brunei_2023_bat_16"]
    dataset_names = ["brunei_2023_bat_test_13_1"]
    indexes = []
    iou_loss_mean = []
    iou_loss_std = []
    f = plt.figure()
    ax = f.add_subplot()
   
   
    for index, dataset_name in enumerate(dataset_names): 
        dataset_root = os.path.join(dataset_path, dataset_name)
        
        mean, std, iou_loss_list = iou_loss(dataset_root)
        indexes.append(index)
        iou_loss_mean.append(mean)
        iou_loss_std.append(std)
        ax.plot(iou_loss_list)
        break
        
        
    
    '''
    print(len(iou_loss_mean))
    print(len(iou_loss_std))
    print(len(indexes))
    plt.xticks(visible=False)
    #plt.xlabel("dataset")
    #plt.ylabel("IOU Loss")
    plt.bar(indexes, iou_loss_mean,edgecolor='grey', color='grey')
    plt.errorbar(indexes, iou_loss_mean, iou_loss_std,ls='none',color='black', elinewidth=1,capthick=1, capsize = 10)
    #plt.show()
    
    indexes = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    frame_number = [219, 601, 156, 347, 500, 681, 866, 361, 300, 236]
    plt.bar(indexes, frame_number,edgecolor='grey', color='grey')
    #ax.yaxis.tick_right()
    plt.savefig("frame_number.svg")
    
    #print(json_file_root)
    print("IOU loss plot")
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    