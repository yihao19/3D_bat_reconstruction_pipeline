# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:47:41 2023

@author: 18505
"""

'''
function using manually labeled joints to validate the accuracy of the reconstruction
in L2 norm
'''


import scipy.io
import os
import json
import sys
import numpy as np
import math
import matplotlib.pyplot as plt 
def validation(frame_index,key_point_index = 9): 
    mat = scipy.io.loadmat('landmarks.mat')
    
    validation_location = mat['LANDMARKS'][0][key_point_index][2]
    print(mat['LANDMARKS'][0][key_point_index][0])
    return validation_location[:, frame_index]

def read_pose_json(json_file):
    
    f = open(json_file)
    output_json = json.load(f)
    
    pose_array = output_json['joints_tail']
    pose_array = np.array(pose_array)[0]

    joint_location = pose_array[31]
    length_m = np.linalg.norm(pose_array[7] - pose_array[5])
    print(f"length_m: {length_m}")
    
    return joint_location
    

if __name__=="__main__":
    print("3D_validation")
    total_loss = np.zeros(101)
    for index in range(10): 
        loss = np.loadtxt(f"{index}.txt")
        total_loss += loss
    
    total_loss = total_loss / 10
    f = plt.figure()
    ax = f.add_subplot()
    ax.plot(total_loss[20:], color='black')
        
 
