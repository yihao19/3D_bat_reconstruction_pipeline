import numpy as np
import math
import torch
from torch.nn import functional as F
from utils.geometry import batch_rodrigues
import os
import json
import torch

import matplotlib.pyplot as plt
from math import sin, cos
import numpy as np

class LBS():
    '''
    Implementation of linear blend skinning, with additional bone and scale
    Input:
        V (BN, V, 3): vertices to pose and shape
        pose (BN, J, 3, 3) or (BN, J, 3): pose in rot or axis-angle
        bone (BN, K): allow for direct change of relative joint distances
        scale (1): scale the whole kinematic tree
    '''
    def __init__(self, J, parents, weights):
        self.n_joints = J.shape[1]  # J shape (2, number of joint)
        print("J shape: ", J.shape)
        self.h_joints = F.pad(J.unsqueeze(-1), [0,0,0,1], value=0) #
        print("J_shape:", type(self.h_joints.shape))
        print("J root: ", J[:, [0], :].shape)# root joint
        print("J parents: ",parents[1:])
        print("J parents: ", J[:, parents[1:]].shape)
        print("J root - parents: ", (J[:, 1:]-J[:, parents[1:]]).shape) # the rest 24 joint 
        self.kin_tree = torch.cat([J[:,[0], :], J[:, 1:]-J[:, parents[1:]]], dim=1).unsqueeze(-1)
        #
        print("kin_tree: ", self.kin_tree.shape)
        print("parent: ", parents.shape)
        self.parents = parents #
        self.weights = weights.float()
        print(self.weights)
        print(weights.shape)
        
    def __call__(self, V,J, pose, to_rotmats=True):
        batch_size = len(V)
        device = pose.device
        V = F.pad(V.unsqueeze(-1), [0,0,0,1], value=1) # turn into the homogeneous coordinates
        J = F.pad(J.unsqueeze(-1), [0,0,0,1], value=1)
        kin_tree = self.kin_tree#(scale*self.kin_tree) * bone[:, :, None, None]
        
        if to_rotmats:
            pose = batch_rodrigues(pose.view(-1, 3))
        pose = pose.view([batch_size, -1, 3, 3])
        T = torch.zeros([batch_size, self.n_joints, 4, 4]).float().to(device)
        T[:, :, -1, -1] = 1
        T[:, :, :3, :] = torch.cat([pose, kin_tree], dim=-1)
        T_rel = [T[:, 0]] # the root transformation matrix
        for i in range(1, self.n_joints):
            print(i, self.parents[i])
            T_rel.append(T_rel[self.parents[i]] @ T[:, i])
        T_rel = torch.stack(T_rel, dim=1)
        T_rel[:,:,:,[-1]] -= T_rel.clone().float() @ (self.h_joints.float())
        T_ = self.weights @ T_rel.view(batch_size, self.n_joints, -1)
        T_ = T_.view(batch_size, -1, 4, 4)
        
        V = T_.float() @ V.float()
        # for bones
        T_ =  T_rel.view(batch_size, self.n_joints, -1)
        T_ = T_.view(batch_size, -1, 4, 4)
        J = T_.float() @ J.float()
        return V[:, :, :3, 0], J[:, :, :3, 0]
    

    
    


