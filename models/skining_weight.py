# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:35:50 2023

@author: 18505
"""

import torch
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

bone_skining_matrix_path='./new_bat_params_version2_forward.pkl'
bone_skining_matrix_path_local_adj='./new_bat_params_version2_forward_local_adj.pkl'
   
output_json = "G:\\PhDProject_real_data\\brunei_2023_bat_test_13_1\\rearrange_pose\\680\\output.json"
    # set template mesh
    # the mesh object no need to change since the vertices will move with the
    # joints
    # put the mesh of the model in rest pose in OBJ file and bone and default skining
    # matrix in the corresponding pkl file
    #self.estimated_location_file = estimated_location_file
    
with open(bone_skining_matrix_path, 'rb') as f:
    data = pickle.load(f)

print(data['weights'].shape)

f = open(output_json)
 
# returns JSON object as 
# a dictionary
updated_data = json.load(f)

print(np.array(updated_data['skining_tensor']).shape)

print(sum(data['weights'][10, :]))

fig=plt.figure()

plt.pcolor(data['weights'].transpose())

plt.colorbar(orientation='vertical')
plt.xlabel("vertex index")
plt.ylabel("Bone index")
plt.title("Original") 
plt.show()

fig=plt.figure()

plt.pcolor(np.array(updated_data['skining_tensor']).transpose())

print(sum(np.array(updated_data['skining_tensor'])[1, :]))

plt.colorbar(orientation='vertical')
plt.xlabel("vertex index")
plt.ylabel("Bone index")
plt.title("Updated") 
plt.show()


