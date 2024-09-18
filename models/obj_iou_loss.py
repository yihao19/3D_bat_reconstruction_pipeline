# -*- coding: utf-8 -*-
"""
Created on Tue May  7 22:55:42 2024

@author: 18505
"""
"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
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
'''
trying to make the model learning euler angle and displacement
by using camera matrix

'''
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, './')

'''
define the dataset, it contains the silhouette images 
and the camera matrix

camera_list: passing the camera index to control the number of passed camera
and angles: only read the camera meta that the silouette contains part of the bat

'''
class image_dataset(Dataset):
    def __init__(self, camera_meta_path, camera_list_path, silouette_image_path, current_pose, use_previous):
        self.camera_meta_path = camera_meta_path
        self.camera_list_path = camera_list_path
        self.silouette_image_path = silouette_image_path
        
        self.use_previous = use_previous
        self.current_pose = current_pose
        # read the file
        camera_list_file = os.path.join(self.camera_list_path, "camera.txt")
        camera_list= []
        camera_list_file = open(camera_list_file)
        camera_list_string = camera_list_file.read()
        camera_list_string = camera_list_string[1: len(camera_list_string)-1]
        camera_list = camera_list_string.split(', ')
        print(camera_list)
        if(len(camera_list) <= 0):
            raise Exception("The number of silouettee image is: {}, not enough images!".format(len(camera_list)))

        self.camera_number = len(camera_list)
        
        entire_camera_matrix = np.loadtxt(os.path.join(self.camera_meta_path, 'camera_meta.txt'))
        self.image_list = []  # store the image path
        
        camera_matrix = np.zeros((self.camera_number, 12))
        counter = 0
        for index in camera_list: 
            index = int(index)
            image_name = "camera{}.png".format(index)
            self.image_list.append(image_name)
            camera_matrix[counter] = entire_camera_matrix[index-1]
            counter += 1
        camera_matrix = np.reshape(camera_matrix, (self.camera_number, 3, 4))
        self.camera_matrix = camera_matrix # camera matrix initialization
        
    def __len__(self):
        return self.camera_number
    def __getitem__(self, idx):
        # return the data sample indicated by the passed index

        index = idx % self.camera_number 

        camera_matrix = self.camera_matrix[index]
        
        image_path = os.path.join(self.silouette_image_path, self.image_list[index])
        
        mask_image = cv.imread(image_path).astype('float32')[:, :, 0] / 255.
        mask_image = np.expand_dims(mask_image, -1)
        #cv.imwrite("test.png",255 * mask_image)
        mask_image = mask_image.transpose((2, 0, 1))
        
        
        if(self.use_previous == False):
           
            prev_pose = 0
            '''
            estimated_location_file = open(os.path.join(self.silouette_image_path, 'estimated_location.txt'))
            estimated_location_string = estimated_location_file.read()
            parts = estimated_location_string.split(' ')
            x_average = float(parts[0])
            y_average = float(parts[1])
            z_average = float(parts[2])
            '''
            estimated_location = 0#np.array([x_average, y_average, z_average]).astype('float32') # randomly assign offset for

        else:
            prev_pose_index = self.current_pose - 1
            current_path = Path(self.silouette_image_path)
            curr_root = current_path.parent
            prev_output_path = os.path.join(curr_root, str(prev_pose_index), "output.json")
            file = open(prev_output_path)
            prev_output = json.load(file) 
            prev_pose = np.array(prev_output['pose'])
            estimated_location = 0#np.array(prev_output['template_displacement']).astype('float32')
        sample = {'mask': mask_image, 
                  'camera_matrix':camera_matrix.astype('float32'), 
                  'prev_pose':prev_pose, 
                  'estimated_location':estimated_location}
        return sample
    
                
def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union ).sum() / intersect.nelement()


def main(test_name, passed_pose_index, epoch, use_previous):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template-mesh', type=str,
                        default=os.path.join(data_dir, './pose150.obj'))
    parser.add_argument('-o', '--image-output-dir', type=str,
                        default=os.path.join(data_dir, 'results/output_deform'))
    args = parser.parse_args()
    
    # make the model data loader style of input 
    os.makedirs(args.image_output_dir, exist_ok=True)

    # start 
    pose_index = int(passed_pose_index)
    image_size = (1024,1280)
    output_path = 'G:/PhDProject_real_data/brunei_{}/rearrange_pose/{}/'.format(test_name, pose_index)
    #camera_list = [1]
    camera_meta_path = 'D:/PhDProject_real_data/brunei_{}/rearrange_pose/'.format(test_name)
    camera_list_path = 'D:/PhDProject_real_data/brunei_{}/rearrange_pose/{}/'.format(test_name, pose_index)
    silouettee_image_path = 'D:/PhDProject_real_data/brunei_{}/rearrange_pose/{}/'.format(test_name, pose_index)
    #estimated_location_file = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/estimated_location.txt'.format(test_name, pose_index)
    args.output_dir  = 'D:/PhDProject_real_data/brunei_{}/reconstruction/'.format(test_name)
    #args.image_output_dir = 'G:/PhDProject_real_data/{}/reconstruction/'.format(test_name)
    
    template_obj_path = "D:/PhDProject_real_data/brunei_{}/reconstruction_v_2.0/2023_bat_15_2_v2_{}.obj".format(test_name, passed_pose_index)
    #template_obj_path = "G:/Users/18505/Desktop/render_images/export_{}.obj".format(pose_index)
    current_pose = pose_index
    # if use_previous == True
    # load the previous pose matrix as a starting point for the current pose reconstruction
    dataset = image_dataset(camera_meta_path, camera_list_path, silouettee_image_path, current_pose, use_previous)
    
    batch_size =dataset.camera_number 
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    template_mesh = sr.Mesh.from_obj(template_obj_path, load_texture=False, texture_res = 5, texture_type='surface')
    
    
    template_mesh.save_obj("test_obj.obj")

    for training_sample in train_dataloader:
  
        images_gt = training_sample['mask'].cuda()
        camera_matrix = training_sample['camera_matrix'].cuda()
        prev_pose = training_sample['prev_pose'].cuda()
        
        estimated_location = training_sample['estimated_location'].cuda()
        #images_gt = torch.from_numpy(images).cuda()
        # at the begining, train the model orientation first

        #mesh
        mesh = sr.Mesh(template_mesh.vertices.repeat(batch_size, 1, 1),template_mesh.faces.repeat(batch_size, 1, 1))
        
        
        renderer = sr.SoftRenderer(image_height=image_size[0], image_width=image_size[1],sigma_val=1e-6,
                                   camera_mode='projection', P = camera_matrix ,orig_height=image_size[0], orig_width=image_size[1], 
                                   near=0, far=100)
        
        # check the mesh vertices and the projection          
        images_pred = renderer.render_mesh(mesh)
        image = images_pred.detach().cpu().numpy()[0].transpose((1 , 2, 0))
        cv.imwrite("test_image.png", 255*image)
        # optimize mesh with silhouette reprojection error and
        # geometry constraints
        # silhouette image predicted will in the 4th element of the vector 
        #print("pred_image shape: ", images_pred.shape)
        IOU_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])
        return IOU_loss.cpu().detach().numpy()

    

if __name__ == '__main__':
    start_pose = 3400
    end_pose = 3658
    epoch = 1
    interval = 1
    test_name = "2023_bat_15_2"
    loss = 0
    loss_list = []
    for pose_index in range(start_pose, end_pose + interval, interval):
        print("working on pose: ", pose_index)
        if(pose_index < 10): 
            pass_pose_index = f"000{pose_index}"
        elif(pose_index < 100): 
            pass_pose_index = f"00{pose_index}"
        elif(pose_index < 1000): 
            pass_pose_index = f"0{pose_index}" 
        else: 
            pass_pose_index = str(pose_index)
        # developing the use_previous to provide extra supervision
        loss = main(test_name, pass_pose_index, epoch, use_previous = False)
       
        print(loss)
        loss_list.append(loss)
        pose_index = int(pose_index)
    loss_array = np.array(loss_list)
   
    print(f"Mean: {np.mean(loss_array)}   std: {np.std(loss_array)}")
    