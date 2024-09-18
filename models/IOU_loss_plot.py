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
from scipy import stats
def iou_loss(dataset_root):
    failed_iou_loss = []
    good_iou_loss = []
    # loop throught the reconstruction and store the name of the folder
    folder_name_list = []
    reconstruction_path = os.path.join(dataset_root, 'reconstruction')
    for reconstruction_name in os.listdir(reconstruction_path): 
        obj_name = reconstruction_name.split('.')[0]
        folder_name = obj_name.split('_')[-1]
        folder_name_list.append(folder_name)
        
    json_file_root = os.path.join(dataset_root, 'rearrange_pose')
    for folder in os.listdir(json_file_root):
        if(folder in folder_name_list):
            continue
        try: 
            json_file = os.path.join(json_file_root, folder, "output.json")
            file = open(json_file)
            data = json.load(file)
            failed_iou_loss.append(data['IOU'])
           
        except: 
            continue
    for folder in os.listdir(json_file_root):
         if(folder not in folder_name_list):
             continue
         try: 
             json_file = os.path.join(json_file_root, folder, "output.json")
             file = open(json_file)
             data = json.load(file)
             good_iou_loss.append(data['IOU'])
            
         except: 
             continue
    
    return good_iou_loss,failed_iou_loss




if __name__ == "__main__": 
    dataset_path = "D:\\PhDProject_real_data\\"
    dataset_names = ["brunei_2023_bat_test_13_2", "brunei_2023_bat_test_13_1", 'brunei_2023_bat_test_14_1',
                     "brunei_2023_bat_test_15_1","brunei_2023_bat_test_16_1", "brunei_2023_bat_test_17_1", 
                     "Brunei_2023_bat_14_1", "Brunei_2023_bat_15_1","Brunei_2023_bat_15_2","Brunei_2023_bat_16"]
    indexes = []
    iou_loss_mean = []
    iou_loss_std = []
    f = plt.figure()
    ax = f.add_subplot()
    good_total_iou_loss = []
    failed_total_iou_loss = []

    for index, dataset_name in tqdm(enumerate(dataset_names)): 
        dataset_root = os.path.join(dataset_path, dataset_name)
        
        good_iou_loss, failed_iou_loss= iou_loss(dataset_root)
        indexes.append(index)
        iou_loss_mean.append(np.mean(good_iou_loss))
        iou_loss_std.append(np.std(good_iou_loss))
        
        good_total_iou_loss =  good_total_iou_loss + good_iou_loss
        failed_total_iou_loss = failed_total_iou_loss + failed_iou_loss
    print("\n")
    print("good_average mean: ", np.mean(np.array(good_total_iou_loss)), "  std: ", np.std(good_total_iou_loss))
    print("failed_average mean: ", np.mean(np.array(failed_total_iou_loss)), "  std: ", np.std(failed_total_iou_loss))
   
    

    plt.xticks(visible=False)
    #plt.xlabel("dataset")
    #plt.ylabel("IOU Loss")
    plt.bar(indexes, iou_loss_mean,edgecolor='grey', color='grey')
    plt.errorbar(indexes, iou_loss_mean, iou_loss_std,ls='none',color='black', elinewidth=1,capthick=1, capsize = 10)
    plt.savefig("all_IOU_loss.svg")
    #plt.show()
    '''
    indexes = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    frame_number = [219, 601, 156, 347, 500, 681, 866, 361, 300, 236]
    plt.bar(indexes, frame_number,edgecolor='grey', color='grey')
    #ax.yaxis.tick_right()
    plt.savefig("frame_number.svg")
    
    #print(json_file_root)
    print("IOU loss plot")
    '''
    plt.plot(good_total_iou_loss)
    plt.plot(failed_total_iou_loss)
    plt.show()
    print(len(good_total_iou_loss), "  ", len(failed_total_iou_loss))
    result = stats.ttest_ind(good_total_iou_loss[:500], good_total_iou_loss)
    print(result)
    result = stats.ttest_ind(failed_total_iou_loss,good_total_iou_loss[:500])
    print(result)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    