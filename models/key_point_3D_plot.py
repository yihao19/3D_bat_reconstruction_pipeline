# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:37:24 2024

@author: yihao
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:24:51 2024

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
import numpy as np
import math
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import pickle
import imageio
import argparse
import math
from torch.utils.data import Dataset
import soft_renderer as sr
import cv2 as cv
import math
from torch import sin, cos
from torch.autograd import Variable
from torch.utils.data import DataLoader
from LBS import LBS
from torch.nn import functional as F
from pathlib import Path
import json
def validation(frame_index,key_point_index = 1): 
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
    return joint_location
    

if __name__=="__main__":
     print("3D_validation")
     
     path = "D:\\PhDProject_real_data\\" 
     test_case = "brunei_2023_bat_test_13_2"
     output_path = os.path.join(path, test_case, "rearrange_pose")
     start_index = 200
     end_index =   300
     valid_frame_counter = 0
     total_distance = 0
     for counter in range(start_index, end_index+1): 
         json_path = os.path.join(output_path, str(counter), "output.json")
         joint_location = read_pose_json(json_path)
        
         frame_index = counter - 1
         
         validation_location = validation(frame_index)
         
         
         if(math.isnan(validation_location[0])): 
             continue

            
         template_obj_path = "D:/PhDProject_real_data/2023_validation/export_0{}.obj".format(counter)
         
         template_mesh = sr.Mesh.from_obj(template_obj_path, load_texture=False, texture_res = 5, texture_type='surface')
         vertices = template_mesh.vertices.cpu().detach().numpy()[0]
         joint_location = vertices[262,:]
         print(counter, " ",100* 4.8 * np.linalg.norm(joint_location - validation_location))
         total_distance += 100* 4.8 * np.linalg.norm(joint_location - validation_location)
         valid_frame_counter += 1
         #print(distance)
     
    
     print(total_distance * 1.0 / valid_frame_counter)
 
