# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:30:50 2023

@author: 18505
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:11:45 2023

@author: 18505
"""
import numpy as np

import cv2 as cv
import pickle
import torch
# read the data template and project the vertices to the image 
# plane and  see if the camera matrix is right
bone_skining_matrix_path = './params.pkl'
with open(bone_skining_matrix_path, 'rb') as f:
     data = pickle.load(f)

thefile = open('test.obj', 'w')
for item in data['v_template']:
  thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))


for item in data['faces']:
  thefile.write("f {0} {1} {2}\n".format(item[0],item[1],item[2]))  

thefile.close()


# check if the joint and the coordinates in the same orientation









