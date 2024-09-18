# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 20:02:23 2023

@author: 18505
"""

'''
generate output for Bokun
'''

# load the template first the get the initial vertices and face first
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse
import torch
import torch.nn as nn
import soft_renderer as sr
import tqdm
import cv2 as cv
import torch
import numpy as np
from skimage.io import imread
import pickle
from LBS import LBS
import json
       
class Model():
    def __init__(self,bone_skining_matrix_path='./new_bat_params_forward.pkl', train_skining_matrix = False):
         super(Model, self).__init__()
         # set template mesh
         # the mesh object no need to change since the vertices will move with the
         # joints
         # put the mesh of the model in rest pose in OBJ file and bone and default skining
         # matrix in the corresponding pkl file
         #self.estimated_location_file = estimated_location_file
         
        
         with open(bone_skining_matrix_path, 'rb') as f:
             data = pickle.load(f)
         # generate the .obj file from params.pkl
    
         self.vertices = torch.tensor(data['v_template']).float().unsqueeze(0).cuda()
         self.faces = torch.tensor(data['faces']).unsqueeze(0).cuda()
         #self.template_mesh.face_vertices =  
         #print(self.template_mesh.face_vertices)
         #self.template_mesh.faces = torch.tensor(data['faces']).unsqueeze(0).cuda()
         joints = data['joints_matrix'][0:3, :].transpose()
         
    
         self.joint_number = joints.shape[0]
         skining = data['weights']
         #kintree_table = data['kintree']  # numpy array that define the kinematic tree of the skeleton
         if(train_skining_matrix == False):
             # use the default 
             skining_tensor  = torch.tensor(skining).unsqueeze(0).cuda()
             self.skining = skining_tensor
         else: 
             # make skinging_tensor a trainable parameters
             print("training with skining matrix...developing later")
         # importing the bones and skining matrix of a bat model
         # make the skining matrix the registered param
         # and joints the registered param
         # first, test the 
         joints_tensor = torch.tensor(joints).unsqueeze(0).cuda()
         self.joints = joints_tensor
         # define the kintree of the skeleton of the bat
         # define in the Blender 
         kintree_table = np.array([[ -1, 0, 1, 1, 3, 4, 5, 4, 7, 4, 9,  1,  11, 12, 13, 12, 15, 12, 17, 0,  19, 0,  21],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]])
    
         self.kintree_table = torch.tensor(kintree_table).cuda()
         self.parents = self.kintree_table[0].type(torch.LongTensor)
         self.LBS_model = LBS(self.joints, self.parents, self.skining)# define the LBS model
    def process(self, f):
        # open the json file and load the pose as tensor
        _dict = json.load(f)
        pose = _dict["pose"]
        vertices_mean = _dict['vertices_mean']
        IOU_loss = _dict['IOU']
        pose = np.array(pose)
        pose = torch.tensor(pose).cuda()
        vertices, joints = self.LBS_model(self.vertices,self.joints, pose, to_rotmats=True)
        joints = joints.cpu().squeeze(0).numpy()
        _dict["joints"] = joints.tolist()
        f.seek(0)
        json.dump(_dict, f)
        f.truncate()
        return joints, vertices_mean, IOU_loss
    
     
     
if __name__=="__main__":
    print("Hello")
    '''
    model = Model()
    total_joints = []
    IOU_losses = []
    vertices_means = []
    for counter in range(600, 1300):
        json_path = "G:\\PhDProject_real_data\\brunei_2023_bat_test_13_1\\rearrange_pose\\{}\\output.json".format(counter)
        
        f = open(json_path, 'r+')
        joints, vertices_mean, IOU_loss = model.process(f)
  
        total_joints.append(joints)
        IOU_losses.append(IOU_loss)
        vertices_means.append(vertices_mean)
    total_joints = np.array(total_joints)
    IOU_losses = np.array(IOU_losses)
    vertices_means = np.array(vertices_means)
    np.save("600_poses_head.npy", total_joints)
    np.save("600_IOU_losses.npy", IOU_losses)
    np.save("600_vertices_means.npy", vertices_means)
    '''
    loss = np.load("600_IOU_losses.npy")
    print(loss.shape)
    plt.plot(loss)
    plt.title("test13_1:straight flight, using previous pose, start from pose 0")
    plt.ylabel("IOU loss")
    plt.xlabel("frame index")
    plt.show()
    
    
    