# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:40:13 2024

@author: 18505
"""

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
def validation(frame_index,key_point_index = 0): 
    mat = scipy.io.loadmat('landmarks.mat')
    
    validation_location = mat['LANDMARKS'][0][key_point_index][2]
    print(mat['LANDMARKS'][0][key_point_index][0])
    return validation_location[:, frame_index]

def read_pose_json(json_file):
    
    f = open(json_file)
    output_json = json.load(f)
    
    pose_array = output_json['joints_tail']
    pose_array = np.array(pose_array)[0]

    joint_location = pose_array[7]
    length_m = np.linalg.norm(pose_array[7] - pose_array[5])
    print(f"length_m: {length_m}")
    
    return joint_location
    

if __name__=="__main__":
    frame_index = 200
    
    points = []
    
    for keypoint_index in range(10): 
        points.append(validation(frame_index, keypoint_index))
   
    
    points = np.array(points)
    print(points.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
         
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
