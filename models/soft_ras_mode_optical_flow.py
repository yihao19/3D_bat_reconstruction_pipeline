# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 21:07:55 2023

@author: 18505
"""

"""
model soft_ras_optical_flow

input: a sequence of silouette images
       sequence of optical flow from only one selected camera
       camera_sequence
training_parameters:
        pose tensor for each frame
shared_parameters: 
        scale for template
        local vertex adjustment for the template
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
'''
function will generate optical flow and return a list optical flow

image_root_path: G:\PhDProject_real_data\brunei_2023_bat_test_13_1\
'''

def generate_optical_flow(start_index, end_index, camera_index, image_root_path):
        
    y_flow_list = []  # len should be end_index - start_index
    x_flow_list = []
    
    camera_folder_path = os.path.joint(image_root_path, "camera{}".format(camera_index))
    
    for index in range(start_index, end_index):
        first_image_path = os.path.join(camera_folder_path, "camera{}{}.png".format(camera_index, start_index))
        second_image_path = os.path.join(camera_folder_path, "camera{}{}.png".format(camera_index, start_index + 1))
        first_image = cv.imread(first_image_path,cv.IMREAD_GRAYSCALE)
        second_image = cv.imread(second_image_path, cv.IMREAD_GRAYSCALE)
        optical_flow = cv.calcOpticalFlowFarneback(first_image, second_image, 
                                               None,
                                               0.5, 3,20, 3, 5, 1.2, 0)
        y_flow_list.append(optical_flow[1])
        x_flow_list.append(optical_flow[0])
    
    return x_flow_list, y_flow_list


'''
given two consecutive frame, using softras to render the optical flow
input should be the first_mesh and second_mesh, 
and pass the render 
'''

def optical_flow_render(first_mesh, second_mesh, camera_matrix, renderer):
    
    #images = renderer(mesh.vertices, mesh.faces, mesh.textures, texture_type='vertex')[0,0:3,:,:].permute(1, 2, 0)
    # assign the displacement to the first mesh as vertex color
    
    flow_mesh =  sr.Mesh(torch.cat([first_mesh.vertices, first_mesh.vertices]), torch.cat([first_mesh.faces,first_mesh.faces]), torch.cat([first_mesh.vertices,second_mesh.vertices]), texture_type='vertex')
    #second_mesh = sr.Mesh(second_mesh.vertices, second_mesh.faces, second_mesh.vertices, texture_type='vertex')
    
    images = renderer(flow_mesh.vertices, flow_mesh.faces, flow_mesh.textures, texture_type='vertex')#[0,0:3,:,:].permute(1, 2, 0)
    #second_image = renderer(second_mesh.vertices, second_mesh.faces, second_mesh.textures, texture_type='vertex')[0,0:3,:,:].permute(1, 2, 0)

    first_image = images[0, 0:3, :, :].permute(1, 2, 0)
    second_image = images[1, 0:3, :, :].permute(1, 2, 0)
    
    first_image = torch.cat([first_image, torch.ones(1024, 1280, 1).cuda()], dim=2)
    second_image = torch.cat([second_image, torch.ones(1024, 1280, 1).cuda()], dim=2)
    
    cam = torch.tensor(camera_matrix.squeeze(0).transpose()).cuda()
    pro_first_image = torch.matmul(first_image, cam)
    pro_second_image = torch.matmul(second_image,cam)
    
    pro_first_image[...,0] = pro_first_image[..., 0] / pro_first_image[..., 2]
    pro_first_image[...,1] = pro_first_image[..., 1] / pro_first_image[..., 2]
    
    pro_second_image[...,0] = pro_second_image[..., 0] / pro_second_image[..., 2]
    pro_second_image[...,1] = pro_second_image[..., 1] / pro_second_image[..., 2]
    
    fw_flow = pro_second_image - pro_first_image
        
    return fw_flow

'''
dataset class: 
    input a sequence of silouettee images from different cameras 
    input a sequence of original images for calculating optical flow 
    input of a sequence of camera matrix 
    output a sequnce of estimated location
    
    # this model doesn't have to use the previous pose for as a reference
'''
class image_dataset(Dataset):
    def __init__(self, camera_meta_path, silouette_image_path,original_image_path, start_pose, end_pose, optical_flow_camera_index):
        self.camera_meta_path = camera_meta_path
        self.silouette_image_path = silouette_image_path
        self.original_image_path = original_image_path
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.optical_flow_camera_index = optical_flow_camera_index
        # initialize the outputs: 
        self.silouette_images_seq = []
        self.original_image_seq = []
        self.estimated_location_seq = []
        self.camera_matrix_seq = []   # since for each pose, the camera index can be different
        # read the file
        entire_camera_matrix = np.loadtxt(os.path.join(self.camera_meta_path, 'camera_meta.txt'))
        self.entire_camera_matrix = entire_camera_matrix
        '''
        camera_list_file = os.path.join(self.camera_list_path, "camera.txt")
        camera_list= []
        camera_list_file = open(camera_list_file)
        camera_list_string = camera_list_file.read()
        camera_list_string = camera_list_string[1: len(camera_list_string)-1]
        camera_list = camera_list_string.split(', ')
        if(len(camera_list) <= 0):
            raise Exception("The number of silouettee image is: {}, not enough images!".format(len(camera_list)))

        self.camera_number = len(camera_list)
        
      
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
        '''
    def __len__(self):
        return 1
    '''
    calculate the optical flow by using the passed original images sequence
    '''
    def calculate_optical_flow(self):
        return
    def __getitem__(self, idx):
        # return the data sample indicated by the passed index
        # load the sequence of the silouette images for each pose
        
        for index in range(self.start_pose, self.end_pose + 1): 
            silouette_path = os.path.join(self.silouette_image_path, str(index))
            # load silouette images path
            camera_list_file = os.path.join(silouette_path, "camera.txt")
            camera_list= []
            camera_list_file = open(camera_list_file)
            camera_list_string = camera_list_file.read()
            camera_list_string = camera_list_string[1: len(camera_list_string)-1]
            camera_list = camera_list_string.split(', ')
            current_image_list = []
            current_camera_matrix_list = []
            for camera_index in camera_list: 
                image_path = os.path.join(silouette_path, "camera{}.png".format(camera_index))
                current_image = cv.imread(image_path).astype('float32')[:, :, 0] / 255.
                current_image_list.append(current_image)
                current_camera_matrix_list.append(self.entire_camera_matrix[index-1])
            # convert the current_image_list and current_camera_matrix_list into the
            current_images = np.asarray(current_image_list)
            current_camera_matrices = np.asarray(current_camera_matrix_list)
            self.silouette_images_seq.append(current_images)
            self.camera_matrix_seq.append(current_camera_matrices)
            
            # load the original images
            
            original_image_path = os.path.join(self.original_image_path,"camera{}".format(self.optical_flow_camera_index), "camera{}{}.png".format(self.optical_flow_camera_index, index))
           
            print(original_image_path)
            original_image = cv.imread(original_image_path)
            self.original_image_seq.append(original_image)
            # load the estimated_location
            estimated_location_file = open(os.path.join(self.silouette_image_path, str(index), 'estimated_location.txt'))
            estimated_location_string = estimated_location_file.read()
            parts = estimated_location_string.split(' ')
            x_average = float(parts[0])
            y_average = float(parts[1])
            z_average = float(parts[2])
            estimated_location = np.array([x_average, y_average, z_average]).astype('float32') # randomly assign offset for
            #
            self.estimated_location_seq.append(estimated_location)
        
        
        
        print(len(self.silouette_images_seq))
        print(len(self.camera_matrix_seq))
        print(len(self.original_image_seq))
        print(len(self.estimated_location_seq))
        print(self.silouette_images_seq[0].shape)
        print(self.camera_matrix_seq[0].shape)
        print(self.original_image_seq[0].shape)
        sample = {'mask_seq': self.silouette_images_seq,
                  'camera_matrix_seq':self.camera_matrix_seq, 
                  'original_images_seq':self.original_image_seq,
                  'estimated_location_seq':self.estimated_location_seq}
        return sample
    
'''
param: template_obj_path: path for template of model in rest pose(obj file)
       bone_skining_matrix_path: path for self-designed bone and default skining_matrix
       joint_list: determine which bone's rotation matrix that you want to trained to get
       train_skining: determine whether you want to train the skining matrix or using 
                      default matrix as hyper-params
'''
class Model(nn.Module):
    def __init__(self, template_obj_path,bone_skining_matrix_path='./new_bat_params_forward.pkl', train_skining_matrix = False):
        super(Model, self).__init__()

        # set template mesh
        # the mesh object no need to change since the vertices will move with the
        # joints
        # put the mesh of the model in rest pose in OBJ file and bone and default skining
        # matrix in the corresponding pkl file
        #self.estimated_location_file = estimated_location_file
        
        self.template_mesh = sr.Mesh.from_obj(template_obj_path, load_texture=True, texture_res = 5, texture_type='surface')
        with open(bone_skining_matrix_path, 'rb') as f:
            data = pickle.load(f)
        # generate the .obj file from params.pkl

        self.template_mesh.vertices = torch.tensor(data['v_template']).float().unsqueeze(0).cuda()
        self.template_mesh.faces = torch.tensor(data['faces']).unsqueeze(0).cuda()
        #self.template_mesh.face_vertices =  
        #print(self.template_mesh.face_vertices)
        #self.template_mesh.faces = torch.tensor(data['faces']).unsqueeze(0).cuda()
        joints = data['joints_matrix'][3:, :].transpose()
        

        self.joint_number = joints.shape[0]
        skining = data['weights']
        #kintree_table = data['kintree']  # numpy array that define the kinematic tree of the skeleton
        if(train_skining_matrix == False):
            # use the default 
            skining_tensor  = torch.tensor(skining).unsqueeze(0).cuda()
            self.register_buffer('skining',skining_tensor)
        else: 
            # make skinging_tensor a trainable parameters
            print("training with skining matrix...developing later")
        # importing the bones and skining matrix of a bat model
        # make the skining matrix the registered param
        # and joints the registered param
        # first, test the 
        joints_tensor = torch.tensor(joints).unsqueeze(0).cuda()
        # define the kintree of the skeleton of the bat
        # define in the Blender 
        kintree_table = np.array([[ -1, 0, 1, 1, 3, 4, 5, 4, 7, 4, 9,  1,  11, 12, 13, 12, 15, 12, 17, 0,  19, 0,  21],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]])
        # define the pose matrix for the joints in the passing list
 
        # empty pose for all bones
        
         
    
        #cuda all the parameters
        #trainable parameters 
        self.kintree_table = torch.tensor(kintree_table).cuda()
        self.register_buffer('joints',joints_tensor)
        self.register_buffer('vertices', self.template_mesh.vertices)
        self.register_buffer('faces', self.template_mesh.faces)
        self.parents = self.kintree_table[0].type(torch.LongTensor)
        self.LBS_model = LBS(self.joints, self.parents, self.skining)# define the LBS model
        self.vertices_number = self.template_mesh.num_vertices
        # optimize for displacement of the center of the mesh ball
        self.register_parameter('displacement', nn.Parameter(torch.zeros(1,1,3)))
        self.register_parameter('scale', nn.Parameter(torch.ones(1)))
        self.register_parameter('local_adjust', nn.Parameter(torch.zeros(1, self.vertices_number, 3))) # apply a small local adjustment on the template 
                                                                                                       # to adjust the template
        
        #self.register_parameter('pitch', nn.Parameter(torch.zeros(1)))
        #self.register_parameter('yaw', nn.Parameter(torch.zeros(1)))
        #self.register_parameter('roll', nn.Parameter(torch.zeros(1)))
        
        self.register_parameter('joint_0',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_1',nn.Parameter(torch.zeros(1, 3),requires_grad=False))
        self.register_parameter('joint_2',nn.Parameter(torch.zeros(1, 3),requires_grad=False)) # 3 DOF
        self.register_parameter('joint_3',nn.Parameter(torch.zeros(1, 3)))    # 1 DOF Z
        self.register_parameter('joint_4',nn.Parameter(torch.zeros(1, 3),requires_grad=False)) # 3 DOF
        self.register_parameter('joint_5',nn.Parameter(torch.zeros(1, 3),requires_grad=False))    # 1 DOF
        self.register_parameter('joint_6',nn.Parameter(torch.zeros(1, 3),requires_grad=False)) # 3 DOF
        self.register_parameter('joint_7',nn.Parameter(torch.zeros(1, 3),requires_grad=False)) # 3 DOF
        self.register_parameter('joint_8',nn.Parameter(torch.zeros(1, 3),requires_grad=False))    # 1 DOF Z
        self.register_parameter('joint_9',nn.Parameter(torch.zeros(1, 3),requires_grad=False))    # 1 DOF Z
        self.register_parameter('joint_10',nn.Parameter(torch.zeros(1, 3),requires_grad=False))# 3 DOF
        self.register_parameter('joint_11',nn.Parameter(torch.zeros(1, 3)))   # 1 DOF Z
        self.register_parameter('joint_12',nn.Parameter(torch.zeros(1, 3),requires_grad=False))# 3 DOF
        self.register_parameter('joint_13',nn.Parameter(torch.zeros(1, 3),requires_grad=False))   # 1 DOF Z
        self.register_parameter('joint_14',nn.Parameter(torch.zeros(1, 3),requires_grad=False))# 3 DOF
        self.register_parameter('joint_15',nn.Parameter(torch.zeros(1, 3),requires_grad=False))   # 1 DOF Z
        self.register_parameter('joint_16',nn.Parameter(torch.zeros(1, 3),requires_grad=False))# 3 DOF
        self.register_parameter('joint_17',nn.Parameter(torch.zeros(1, 3),requires_grad=False))   # 1 DOF Z
        self.register_parameter('joint_18',nn.Parameter(torch.zeros(1, 3),requires_grad=False))   # 1 DOF Z
        
        # make small displacement of the 
        #self.register_parameter("pose_tensor", nn.Parameter(torch.zeros(1,19,3)))
        # add the pose_tensor of the previous mesh model
        self.pose_tensor = torch.zeros((23, 3)).cuda()
        
        #self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        #self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
        #
    
    '''
    # model's forward function'
    
    '''
    def forward(self, batch_size, estimated_location, use_previous = False, prev_pose = None):
        # define how the vertices is going to change
        '''
        making rotation matrix out of the euler angles
        
        tensor_1 = torch.ones(1).cuda()
        tensor_0 = torch.zeros(1).cuda()
 
        RX = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0]),
                torch.stack([tensor_0, cos(self.roll), -sin(self.roll)]),
                torch.stack([tensor_0, sin(self.roll), cos(self.roll)])]).reshape(3,3).cuda()

        RY = torch.stack([
                        torch.stack([cos(self.pitch), tensor_0, sin(self.pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-sin(self.pitch), tensor_0, cos(self.pitch)])]).reshape(3,3).cuda()
        
        RZ = torch.stack([
                        torch.stack([cos(self.yaw), -sin(self.yaw), tensor_0]),
                        torch.stack([sin(self.yaw), cos(self.yaw), tensor_0]),
                        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3).cuda()
        
        
        rotation =torch.mm(RX, torch.mm(RY, RZ))
        rotation = torch.unsqueeze(rotation, dim=0)
        '''
        # add the displacement first
        
        # rotation the model next
        # rotate the template vertices and the skeleton
        
 
      
        #vertices = torch.bmm(self.vertices,rotation) 
        #joints = torch.bmm(self.joints, rotation.double())
        
        
        
        
        # figure out the joint rotation matrix third
        #np.savetxt('./test_1.txt', vertices.detach().cpu().numpy()[0])
        
        # using sigmoid / tanh function to limit the rotation degree
        # limit the bone rotation to 45 degree maximum pi/4
        # sigmoid x - > (0, 1) 
        # tanh    x - > (-1, 1)
        # assign all bones with fill DOF but a limited rotation angle
        # body rotation matrix
        
        if(use_previous == False):
            self.pose_tensor[0][:] = math.pi / 2 * torch.tanh(self.joint_0)
            self.pose_tensor[1][:] = math.pi / 4 * torch.tanh(self.joint_1)
            self.pose_tensor[2][:] = math.pi / 9 * torch.tanh(self.joint_2)  # make the neck bone trainable(slightly)
            # for the shoulder using the symmetric deformation
            self.pose_tensor[3][0] = math.pi / 3 * torch.tanh(self.joint_3[0][0])
            self.pose_tensor[3][1] = math.pi / 3 * torch.tanh(self.joint_3[0][1])
            self.pose_tensor[3][2] = math.pi / 3 * torch.tanh(self.joint_3[0][2])
            
            
            self.pose_tensor[11][0] = math.pi / 3 * torch.tanh(self.joint_3[0][0])
            self.pose_tensor[11][1] = -math.pi / 3 * torch.tanh(self.joint_3[0][1])
            self.pose_tensor[11][2] = -math.pi / 3 * torch.tanh(self.joint_3[0][2])
            
            self.pose_tensor[4][:] = math.pi / 3 * torch.tanh(self.joint_4)
            self.pose_tensor[12][:] = math.pi / 3 * torch.tanh(self.joint_12)
            
            #self.pose_tensor[14][:] = math.pi / 4 * torch.tanh(self.joint_14)
            #self.pose_tensor[10][:] = math.pi / 4 * torch.tanh(self.joint_10)
            #self.pose_tensor[6][:] = math.pi / 4 * torch.tanh(self.joint_6)
            #self.pose_tensor[7][:] = math.pi / 4 * torch.tanh(self.joint_7)
            #self.pose_tensor[16][:] = math.pi / 4 * torch.tanh(self.joint_16)
            # since the template is fully streched, some angle value can only be negative
            self.pose_tensor[13][:] = math.pi / 3 * torch.tanh(self.joint_13)
            self.pose_tensor[5][:] = math.pi / 3 * torch.tanh(self.joint_5)
        else: 
            
            # use the previous pose and limit the angle, since it start from the previous
            self.pose_tensor[0][:] = prev_pose[0][:] + math.pi / 2 * torch.tanh(self.joint_0)
            self.pose_tensor[1][:] = prev_pose[1][:] + math.pi / 4 * torch.tanh(self.joint_1)
            self.pose_tensor[2][:] = prev_pose[2][:] + math.pi / 9 * torch.tanh(self.joint_2)  # make the neck bone trainable(slightly)
            # for the shoulder using the symmetric deformation
            self.pose_tensor[3][0] = prev_pose[3][0] + math.pi / 3 * torch.tanh(self.joint_3[0][0])
            self.pose_tensor[3][1] = prev_pose[3][1] + math.pi / 3 * torch.tanh(self.joint_3[0][1])
            self.pose_tensor[3][2] = prev_pose[3][2] + math.pi / 3 * torch.tanh(self.joint_3[0][2])
            
            
            self.pose_tensor[11][0] = prev_pose[11][0] + math.pi / 3 * torch.tanh(self.joint_3[0][0])
            self.pose_tensor[11][1] = prev_pose[11][1] -math.pi / 3 * torch.tanh(self.joint_3[0][1])
            self.pose_tensor[11][2] = prev_pose[11][2] -math.pi / 3 * torch.tanh(self.joint_3[0][2])
            
            self.pose_tensor[4][:] = prev_pose[4][:] + math.pi / 3 * torch.tanh(self.joint_4)
            self.pose_tensor[12][:] = prev_pose[12][:] + math.pi / 3 * torch.tanh(self.joint_12)
            
            #self.pose_tensor[14][:] = math.pi / 4 * torch.tanh(self.joint_14)
            #self.pose_tensor[10][:] = math.pi / 4 * torch.tanh(self.joint_10)
            #self.pose_tensor[6][:] = math.pi / 4 * torch.tanh(self.joint_6)
            #self.pose_tensor[7][:] = math.pi / 4 * torch.tanh(self.joint_7)
            #self.pose_tensor[16][:] = math.pi / 4 * torch.tanh(self.joint_16)
            # since the template is fully streched, some angle value can only be negative
            self.pose_tensor[13][:] = prev_pose[13][:] + math.pi / 3 * torch.tanh(self.joint_13)
            self.pose_tensor[5][:] = prev_pose[5][:]  + math.pi / 3 * torch.tanh(self.joint_5)
            
        
        '''
        '''
        '''
        self.pose_tensor[8][2] = self.joint_8
        self.pose_tensor[9][2] = self.joint_9
       
        self.pose_tensor[11][2] = self.joint_11
       
        self.pose_tensor[13][2] = self.joint_13
        
        self.pose_tensor[15][2] = self.joint_15
        
        self.pose_tensor[17][2] = self.joint_17
        '''
        #self.pose_tensor[1] = self.joint_0
        
        #self.pose_tensor = self.pose_tensor.unsqueeze(0)
        # model will deform the mesh and then add the predetermined offset and learned displacement
        # apply the small adjustment on template first
        
        vertices = self.vertices + 0.1 * torch.tanh(self.local_adjust.cuda())
        vertices, joints = self.LBS_model(vertices,self.joints, self.pose_tensor, to_rotmats=True)
        #self.pose_tensor = self.pose_tensor.squeeze()
        
        estimated_location = estimated_location.unsqueeze(dim = 1)
        vertices = 0.0035 * self.scale * vertices + estimated_location[0].repeat(1, self.vertices_number, 1).cuda() + 0.1 * torch.tanh(self.displacement.repeat(1, self.vertices_number, 1)).cuda() 
        joints = 0.0035 * self.scale * joints + estimated_location[0].repeat(1, self.joint_number, 1).cuda() + 0.1 * torch.tanh(self.displacement.repeat(1, self.joint_number, 1)).cuda() 
        #self.vertices = vertices #+ self.random_dis.repeat(1, self.vertices_number, 1).cuda()
        
        #self.joints = joints #+ self.random_dis.repeat(1, self.joint_number, 1).cuda()
        
        #return
        #np.savetxt('./test_2.txt', verts.detach().cpu().numpy()[0])
        
        # apply Laplacian and flatten geometry constraints
        #laplacian_loss = self.laplacian_loss(vertices).mean()
        #flatten_loss = self.flatten_loss(vertices).mean()
        # add l2 regularization for small wing bones
        l2_norm = 0#torch.norm(self.displacement)   
        # define the return package, including the pose_tensor, vertices, faces, joints location, initial displacement, and local positon adjustment
        
        return sr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1)), self.pose_tensor, self.scale, self.displacement, self.local_adjust, vertices, joints
'''
IOU loss define the  
'''                  
def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union ).sum() / intersect.nelement()


def main(test_name, start_pose, end_pose, epoch):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template-mesh', type=str,
                        default=os.path.join(data_dir, './pose150.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
                        default=os.path.join(data_dir, 'results/output_deform'))
    args = parser.parse_args()
    
    # make the model data loader style of input 
    os.makedirs(args.output_dir, exist_ok=True)

 
    # start 
    
    image_size = (1024,1280)
    #output_path = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/'.format(test_name, pose_index)
    #camera_list = [1]
    camera_meta_path = 'G:/PhDProject_real_data/{}/rearrange_pose/'.format(test_name)
    camera_list_path = 'G:/PhDProject_real_data/{}/rearrange_pose/'.format(test_name)
    silouettee_image_path = 'G:/PhDProject_real_data/{}/rearrange_pose/'.format(test_name)
    original_image_path = "G:/PhDProject_real_data/{}/".format(test_name)
    #estimated_location_file = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/estimated_location.txt'.format(test_name, pose_index)
    optical_flow_camera_index = 41
    #current_pose = pose_index
    # if use_previous == True
    # load the previous pose matrix as a starting point for the current pose reconstruction
    #def __init__(self, camera_meta_path, silouette_image_path,original_image_path, start_pose, end_pose, optical_flow_camera_index,use_previous):
    dataset = image_dataset(camera_meta_path, silouettee_image_path,original_image_path, start_pose, end_pose, optical_flow_camera_index )
    
    
    
    

    batch_size = 1
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    
    for data in train_dataloader:
        print(len(data))
    return

    
    #return

    model = Model(args.template_mesh).cuda()
    

    optimizer = torch.optim.Adam(model.parameters(), 0.01,betas=(0.5, 0.99))

    #renderer.transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

    epoch = tqdm.tqdm(list(range(0,epoch)))
    gif_images = []
    writer = imageio.get_writer(os.path.join('./', 'deform_bat_{}.gif'.format(pose_index)), mode='I')
    for i in epoch:
        
        for training_sample in train_dataloader:
      
            images_gt = training_sample['mask'].cuda()
            camera_matrix = training_sample['camera_matrix'].cuda()
            prev_pose = training_sample['prev_pose'].cuda()
            
            estimated_location = training_sample['estimated_location'].cuda()
            #images_gt = torch.from_numpy(images).cuda()
            # at the begining, train the model orientation first
            
            if(i >= 0):
                
                for name, param in model.named_parameters():
                        param.requires_grad = True
                        
            mesh, current_pose, scale, displacement, local_adjust, vertices, joints = model(batch_size, estimated_location, use_previous, prev_pose[0])
            
            
            renderer = sr.SoftRenderer(image_height=image_size[0], image_width=image_size[1],sigma_val=1e-6,
                                       camera_mode='projection', P = camera_matrix,orig_height=image_size[0], orig_width=image_size[1], 
                                       near=0, far=100)
            
            # check the mesh vertices and the projection          
            images_pred = renderer.render_mesh(mesh)
            #print("image_pred shape: ", images_pred.shape)
            # optimize mesh with silhouette reprojection error and
            # geometry constraints
            # silhouette image predicted will in the 4th element of the vector 
            #print("pred_image shape: ", images_pred.shape)
            IOU_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])
            loss = IOU_loss #+ 1 * l2_norm 
            
            pose_loss = torch.tensor(0)

            if(use_previous == True):
                # only the body orientation is considered
                pose_loss = 0.5 * torch.norm(current_pose[:][0] - prev_pose[0][0]) + 0.1 * torch.norm(current_pose[:][1:] - prev_pose[0][1:])
            loss = IOU_loss + pose_loss     
            epoch.set_description('IOU Loss: %.4f   Pose Loss: %.4f' % (IOU_loss.item(),pose_loss.item()))
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        if i % 1 == 0:
            #print("pred_image shape: ",images_pred.detach().cpu().numpy()[0].shape )
            image = images_pred.detach().cpu().numpy()[1].transpose((1 , 2, 0))
            writer.append_data((255*image[:, :, 0]).astype(np.uint8))
            
            for counter in range(1):
                
                image = images_pred.detach().cpu().numpy()[1].transpose((1 , 2, 0))
                imageio.imsave(os.path.join(args.output_dir, 'pred_camera_{}_{}.png'.format(i, counter)), (255*image[..., 1]).astype(np.uint8))
            image_gt = images_gt.detach().cpu().numpy()[0].transpose((1, 2, 0))
            
    output_mesh, current_pose, current_scale, current_displacement, local_adjust, vertices, joints = model(1, estimated_location,use_previous, prev_pose[0])
    output_mesh.save_obj(os.path.join(args.output_dir, '{}_bat_{}.obj'.format(test_name, pose_index)), save_texture=False)
    current_pose = current_pose.cpu().detach().numpy().tolist()
    current_scale = current_scale.cpu().detach().item()
    current_displacement = current_displacement.cpu().detach().numpy().tolist()
    local_adjust = local_adjust.cpu().detach().numpy().tolist()
    vertices = vertices.cpu().detach().numpy()
    vertices_mean = np.mean(vertices[0], axis=0)
    joints = joints.cpu().detach().numpy().tolist()

    save_dict = {"pose":current_pose,
                 "scale":current_scale, 
                 "displacement":current_displacement, 
                 "local_adjust":local_adjust, 
                 "pose_loss":pose_loss.cpu().detach().item(), 
                 "IOU":IOU_loss.cpu().detach().item(),
                 "vertices_mean": vertices_mean.tolist()}
    with open(os.path.join(silouettee_image_path, "output.json"), "w") as output:
        json.dump(save_dict, output)
        
    

if __name__ == '__main__':
    start_pose = 1
    end_pose = 10
    interval = 1
    test_name = "brunei_2023_bat_test_13_2"
   
    epoch = 100
    # developing the use_previous to provide extra supervision
    main(test_name, start_pose, end_pose, epoch)
    
    
    