# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 22:32:16 2023

@author: 18505
"""

import torch
import pickle
import json
import numpy as np

bone_skining_matrix_path='./new_bat_params_version2_forward.pkl'
bone_skining_matrix_path_local_adj='./new_bat_params_version2_forward_local_adj.pkl'
   
output_json = "G:\\PhDProject_real_data\\brunei_2023_bat_test_13_1\\rearrange_pose\\600\\output.json"
    # set template mesh
    # the mesh object no need to change since the vertices will move with the
    # joints
    # put the mesh of the model in rest pose in OBJ file and bone and default skining
    # matrix in the corresponding pkl file
    #self.estimated_location_file = estimated_location_file
    
with open(bone_skining_matrix_path, 'rb') as f:
    data = pickle.load(f)

file = open(output_json)
prev_output = json.load(file) 
local_adj = np.array(prev_output['local_adjust'])[0]
local_adj = 0.05 * np.tanh(local_adj)
print(np.array(prev_output['local_adjust']))
print(data['v_template'].shape)
'''
def save_pkl(path, result):
    """"
    save pkl file
    """
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(path, 'wb') as result_file:
        pickle.dump(result, result_file, protocol=2)
'''
data['v_template'] = data['v_template'] + local_adj

with open(bone_skining_matrix_path_local_adj, 'wb') as result_file:
    pickle.dump(data, result_file, protocol=2)

