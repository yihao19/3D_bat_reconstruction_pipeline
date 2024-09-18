# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:06:20 2024

@author: yihao
"""

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
def iou_loss(json_file_root):
    iou_loss = []
    for folder in os.listdir(json_file_root): 
        try: 
            json_file = os.path.join(json_file_root, folder, "output.json")
            file = open(json_file)
            data = json.load(file)
            iou_loss.append(data['IOU'])
        except: 
            continue
    
    return np.mean(iou_loss), np.std(iou_loss)




if __name__ == "__main__": 
    dataset_path = "D:\\PhDProject_real_data\\"
    dataset_name = "brunei_2023_bat_test_13_2"
    indexes = []
    camera_number_list = []
   
    f = plt.figure()
    ax = f.add_subplot()
   

    
    for index in range(200, 300 + 1, 1): 
        indexes.append(index-200)
        
        txt_file_path = os.path.join(dataset_path, dataset_name, 'rearrange_pose', str(index), 'camera.txt')
        f = open(txt_file_path, "r")
        camera_number =f.read()[1:-1].split(', ')
        camera_number_list.append(len(camera_number))
        
        
    

    plt.xticks(visible=False)
    #plt.xlabel("dataset")
    #plt.ylabel("IOU Loss")
    ax.scatter(indexes, camera_number_list,color='black')
    ax.set_ylim(ymin=0)
    #plt.errorbar(indexes, iou_loss_mean, iou_loss_std,ls='none',color='black', elinewidth=1,capthick=1, capsize = 10)
    #plt.show()

    #indexes = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    #frame_number = [219, 601, 156, 347, 500, 681, 866, 361, 300, 236]
    #plt.bar(indexes, frame_number,edgecolor='grey', color='grey')
    #ax.yaxis.tick_right()
    plt.savefig("camera_number_sequnece_1.svg")
    
    #print(json_file_root)
    print("IOU loss plot")
    