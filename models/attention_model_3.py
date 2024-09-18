"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
from __future__ import print_function, division
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
from torch import sin, cos
from torch.autograd import Variable
from torch.utils.data import DataLoader
from LBS import LBS
from torch.nn import functional as F
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
'''
trying to make the model learning euler angle and displacement
by using camera matrix

'''
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, './')

'''
attention model version 3: 
    using transformer to figure out the best weight for each silouette image
    by reducing the IOU loss of the silouette images
    by reducing the l2 loss of the template and GT pose
    this model will not be using any previous or futuer pose
'''

'''
dataset right now should include the ground truth pose
and the resized original image
'''
class image_dataset(Dataset):
    def __init__(self, camera_meta_path, camera_list_path, silouette_image_path, original_image_path, current_pose):
        self.camera_meta_path = camera_meta_path
        self.camera_list_path = camera_list_path
        self.silouette_image_path = silouette_image_path
        self.original_image_path = original_image_path
        self.current_pose = current_pose # use current_pose to get the original images
        # read the file
        camera_list_file = os.path.join(self.camera_list_path, "camera.txt")
        camera_list= []
        camera_list_file = open(camera_list_file)
        camera_list_string = camera_list_file.read()
        camera_list_string = camera_list_string[1: len(camera_list_string)-1]
        camera_list = camera_list_string.split(', ')
        print(camera_list)
        if(len(camera_list) < 3):
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
    '''
    convert the obj file into xyz point cloud, return the array [# number, 3]
    '''
    def obj_to_xyz(self):
        obj_file_path =  os.path.join(self.silouette_image_path, "reconstruction.obj")
        point_list = []
        obj_file = open(obj_file_path)
        lines = obj_file.readlines()
        for line in lines: 
            parts = line.split(' ')
            if(parts[0] == 'v'):
                point_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
        output = np.array(point_list)
        return output
    
    def __getitem__(self, idx):
        # return the data sample indicated by the passed index
        # camera_name and index dict
        camera_index = {21:0, 23:1, 24:2, 25:3, 31:4, 32:5, 33:6, 34:7, 35:8, 41:9, 42:10, 43:11, 44:12}
        index = idx % self.camera_number 
        camera_matrix = self.camera_matrix[index]
        camera_name = 0
        for key in camera_index: 
            if(camera_index[key] == index):
                camera_name = key
        image_path = os.path.join(self.silouette_image_path, self.image_list[index])
        
        oringinal_image_path = os.path.join(self.original_image_path, "camera{}".format(camera_name), 'camera{}{}.png'.format(camera_name, self.current_pose))
        mask_image = cv.imread(image_path).astype('float32')[:, :, 0] / 255.
        mask_image = np.expand_dims(mask_image, -1)
        #cv.imwrite("test.png",255 * mask_image)
        mask_image = mask_image.transpose((2, 0, 1))
        
        original_image = cv.imread(image_path).astype('float') / 255.
        original_image = cv.resize(original_image, (224, 224))
        
        original_image = original_image.transpose((2, 0, 1))
       
        reconstruction = self.obj_to_xyz()
        sample = {'mask': mask_image, 
                  'camera_matrix':camera_matrix.astype('float32'), 
                  'original':original_image.astype('float'), 
                  'reconstruction':reconstruction}
        return sample

    
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads = 1):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embeded size needs to be div by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # output the same size
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        # value/keys/query shape: (1, #camera_number, # flatten_silouette_pixels)
        # batch size is going to be 1
        # value_len, key_len, query_len = # camera_number
        
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # plist embedding into self.heads pieces
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)
        
        
        # shape: (1, # camera_number, 1, # flatten_silouette_pixel)
        # shape: (1, # camera_number, 1, # flatten_silouette_pixel)
        # shape: (1, # camera_number, 1, # flatten_silouette_pixel)
        values = self.values(values)
        keys = self.keys(keys)
        querys = self.queries(queries)
        
        # what is this statement
        # queries: 1, # camera_number, 1, # flatten_silouette_pixel
        # keys   : 1, # camera_number, 1, # flatten_silouette_pixel
        # output : 1, 1, queries, key (1, 1, 10, 10)
        # interaction between the 10 camera information
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        '''
        if mask != None:
            energy = energy.masked_fill(mask == 0, float("-1e28"))
        '''  
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) 
        # attention shape = (N, heads, query_len, key_len)
        # values shape = (N, value_len, heads, heads_dim)
        # output = (N, query_len, heads, heads_dim)
        # attention shape: (1, 1, 10, 10)
        # values shape: 1, 10, 1, # flatten_silouette_pixel
        # output shape: 1, 10, 1, # flatten_silouette_pixel
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim)
        # output shape: 1, 10, flatten_silouette_pixel
        out = self.fc_out(out)
        # output for the second layer
        return out
'''
define the transformation block including the layer normalization and residule block
'''
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size), 
            nn.ReLU(), 
            nn.Linear(forward_expansion*embed_size, embed_size)
            )
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out
'''
define the transformer encoder to produce the weight for each image
'''
class Encoder(nn.Module):
    def __init__(
            self,
            input_size,
            num_data, 
            embed_size, 
            num_layers, 
            heads, 
            device, 
            forward_expansion, 
            dropout, 
            ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.num_data = num_data
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, 
                                 heads, 
                                 dropout=dropout, 
                                 forward_expansion=forward_expansion
                                 )
                for _ in range(num_layers)
                ]
            )
        self.flatten = nn.Flatten(start_dim = 1)
        self.fc = nn.Linear(self.embed_size * self.num_data, self.num_data)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, embed_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, mask):
        
        out = self.linear(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = self.flatten(out)
        out = self.fc(out)
        
        out = self.softmax(out)
        return out
'''
param: template_obj_path: path for template of model in rest pose(obj file)
       bone_skining_matrix_path: path for self-designed bone and default skining_matrix
       joint_list: determine which bone's rotation matrix that you want to trained to get
       train_skining: determine whether you want to train the skining matrix or using 
                      default matrix as hyper-params
'''
class Model(nn.Module):
    def __init__(self, template_obj_path, estimated_location_file,bone_skining_matrix_path='./new_bat_params.pkl', train_skining_matrix = False):
        super(Model, self).__init__()

        # set template mesh
        # the mesh object no need to change since the vertices will move with the
        # joints
        # put the mesh of the model in rest pose in OBJ file and bone and default skining
        # matrix in the corresponding pkl file
        self.estimated_location_file = estimated_location_file
        
        self.template_mesh = sr.Mesh.from_obj(template_obj_path, load_texture=True, texture_res = 5, texture_type='surface')
        with open(bone_skining_matrix_path, 'rb') as f:
            data = pickle.load(f)
        # generate the .obj file from params.pkl

        self.template_mesh.vertices = torch.tensor(data['v_template']).float().unsqueeze(0).cuda()
        self.template_mesh.faces = torch.tensor(data['faces']).unsqueeze(0).cuda()
        #self.template_mesh.face_vertices =  
        #print(self.template_mesh.face_vertices)
        #self.template_mesh.faces = torch.tensor(data['faces']).unsqueeze(0).cuda()
        joints = data['joints_matrix'][:3, :].transpose()
        

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
        estimated_location_file = open(self.estimated_location_file)
        estimated_location_string = estimated_location_file.read()
        parts = estimated_location_string.split(' ')
        x_average = float(parts[0])
        y_average = float(parts[1])
        z_average = float(parts[2])
        self.random_dis = torch.tensor([x_average, y_average, z_average]) # randomly assign offset for
         
    
        #cuda all the parameters
        #trainable parameters 
        self.kintree_table = torch.tensor(kintree_table).cuda()
        self.register_buffer('joints',joints_tensor)
        self.register_buffer('vertices', self.template_mesh.vertices)
        self.register_buffer('faces', self.template_mesh.faces)
        self.parents = self.kintree_table[0].type(torch.LongTensor)
        self.LBS_model = LBS(self.joints, self.parents, self.skining)# define the LBS model
        #attention_model = Encoder(4096, embed_size, 1, 8, 1, device, 1, 0).to(device)
        self.attention =  Encoder(4096, 9, 16, 1, 1, 'cuda', 128, 0)
        #self.attention = Encoder(64 * 64, 5, 128, 1, 1, 'cuda', 1, 0)
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
        self.register_parameter('joint_1',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_2',nn.Parameter(torch.zeros(1, 3))) # 3 DOF
        self.register_parameter('joint_3',nn.Parameter(torch.zeros(1, 3) ))    # 1 DOF Z
        self.register_parameter('joint_4',nn.Parameter(torch.zeros(1, 3))) # 3 DOF
        self.register_parameter('joint_5',nn.Parameter(torch.zeros(1, 3)))    # 1 DOF
        self.register_parameter('joint_6',nn.Parameter(torch.zeros(1, 3),requires_grad=False)) # 3 DOF
        self.register_parameter('joint_7',nn.Parameter(torch.zeros(1, 3),requires_grad=False)) # 3 DOF
        self.register_parameter('joint_8',nn.Parameter(torch.zeros(1, 3),requires_grad=False))    # 1 DOF Z
        self.register_parameter('joint_9',nn.Parameter(torch.zeros(1, 3),requires_grad=False))    # 1 DOF Z
        self.register_parameter('joint_10',nn.Parameter(torch.zeros(1, 3),requires_grad=False))# 3 DOF
        self.register_parameter('joint_11',nn.Parameter(torch.zeros(1, 3)))   # 1 DOF Z
        self.register_parameter('joint_12',nn.Parameter(torch.zeros(1, 3)))# 3 DOF
        self.register_parameter('joint_13',nn.Parameter(torch.zeros(1, 3)))   # 1 DOF Z
        self.register_parameter('joint_14',nn.Parameter(torch.zeros(1, 3),requires_grad=False))# 3 DOF
        self.register_parameter('joint_15',nn.Parameter(torch.zeros(1, 3),requires_grad=False))   # 1 DOF Z
        self.register_parameter('joint_16',nn.Parameter(torch.zeros(1, 3),requires_grad=False))# 3 DOF
        self.register_parameter('joint_17',nn.Parameter(torch.zeros(1, 3),requires_grad=False))   # 1 DOF Z
        self.register_parameter('joint_18',nn.Parameter(torch.zeros(1, 3),requires_grad=False))   # 1 DOF Z
        
        # make small displacement of the 
        #self.register_parameter("pose_tensor", nn.Parameter(torch.zeros(1,19,3)))
        self.pose_tensor = torch.zeros((23, 3)).cuda()
        
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        #self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
        #
    
    '''
    # model's forward function'
    
    '''
    def forward(self, batch_size, vgg_feature):
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
        self.pose_tensor[0][:] = math.pi / 2 * torch.tanh(self.joint_0)
        self.pose_tensor[1][:] = math.pi / 4 * torch.tanh(self.joint_1)
        # for the shoulder using the symmetric deformation
        self.pose_tensor[3][0] = math.pi / 4 * torch.tanh(self.joint_3[0][0])
        self.pose_tensor[3][1] = math.pi / 4 * torch.tanh(self.joint_3[0][1])
        self.pose_tensor[3][2] = math.pi / 4 * torch.tanh(self.joint_3[0][2])
        
        
        self.pose_tensor[11][0] = math.pi / 4 * torch.tanh(self.joint_11[0][0])
        self.pose_tensor[11][1] = math.pi / 4 * torch.tanh(self.joint_11[0][1])
        self.pose_tensor[11][2] = math.pi / 4 * torch.tanh(self.joint_11[0][2])
        
        self.pose_tensor[4][:] = math.pi / 4 * torch.tanh(self.joint_4)
        self.pose_tensor[12][:] = math.pi / 4 * torch.tanh(self.joint_12)
        
        #self.pose_tensor[14][:] = math.pi / 4 * torch.tanh(self.joint_14)
        #self.pose_tensor[10][:] = math.pi / 4 * torch.tanh(self.joint_10)
        #self.pose_tensor[6][:] = math.pi / 4 * torch.tanh(self.joint_6)
        #self.pose_tensor[7][:] = math.pi / 4 * torch.tanh(self.joint_7)
        #self.pose_tensor[16][:] = math.pi / 4 * torch.tanh(self.joint_16)
        self.pose_tensor[13][:] = math.pi / 4 * torch.tanh(self.joint_13)
        self.pose_tensor[5][:] = math.pi / 4 * torch.tanh(self.joint_5)
        
        
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
        
        vertices = self.vertices + 0.0 * torch.tanh(self.local_adjust.cuda())
        vertices, joints = self.LBS_model(vertices,self.joints, self.pose_tensor, to_rotmats=True)
        #self.pose_tensor = self.pose_tensor.squeeze()
        vertices = 0.0035 * self.scale * vertices + self.random_dis.repeat(1, self.vertices_number, 1).cuda() + 0.1 * torch.tanh(self.displacement.repeat(1, self.vertices_number, 1)).cuda() 
        joints = 0.0035 * self.scale * joints + self.random_dis.repeat(1, self.joint_number, 1).cuda() + 0.1 * torch.tanh(self.displacement.repeat(1, self.joint_number, 1)).cuda() 
        #self.vertices = vertices #+ self.random_dis.repeat(1, self.vertices_number, 1).cuda()
        
        #self.joints = joints #+ self.random_dis.repeat(1, self.joint_number, 1).cuda()
        
        #return
        #np.savetxt('./test_2.txt', verts.detach().cpu().numpy()[0])
        
        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        #flatten_loss = self.flatten_loss(vertices).mean()
        # add l2 regularization for small wing bones
        l2_norm = 0#torch.norm(self.displacement)   
        
        # define the return package, including the pose_tensor, vertices, faces, joints location, initial displacement, and local positon adjustment
        scores = self.attention(vgg_feature, None)
        return sr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1)),  self.pose_tensor, laplacian_loss, scores
'''
IOU loss define the  
'''                  
def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return (1. - (intersect / union).sum() / intersect.nelement())
'''
weighted IOU loss by attention mechanism
score: tensor([# images])
'''
def neg_iou_loss_weighted(predict, target, score):
    dims = tuple(range(predict.ndimension())[1:]) # keep the dimention after the batch
    image_number = score.shape[0]
    extra_IOU = 0
    intersect = (predict * target).sum(dims) # 
    union = (predict + target - predict * target).sum(dims) + 1e-6
    
    original_loss = (1. - (intersect / union).sum() / intersect.nelement())
    
    for counter in range(image_number):
        extra_IOU += (1 - intersect[counter] / union[counter]) * score[counter]

    return original_loss + extra_IOU

def VGG_pretrain():
    vgg16 = models.vgg16_bn().cuda()
    vgg16.load_state_dict(torch.load("./vgg16_bn/vgg16_bn.pth"))
    print(vgg16.classifier[6].out_features) # 1000 
    for param in vgg16.features.parameters():
        param.require_grad = False
    features = list(vgg16.classifier.children())[:-6] # remove the last sequential layer except for the first linear
    # output dim is going to be 4096
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
    return vgg16

def main(test_name, passed_pose_index, epoch, use_previous):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template-mesh', type=str,
                        default=os.path.join(data_dir, './pose150.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
                        default=os.path.join(data_dir, 'results/output_deform'))
    args = parser.parse_args()
    
    # make the model data loader style of input 
    os.makedirs(args.output_dir, exist_ok=True)

 
    # start 
    pose_index = passed_pose_index
    image_size = (1024,1280)
    output_path = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/'.format(test_name, pose_index)
    #camera_list = [1]
    camera_meta_path = 'G:/PhDProject_real_data/{}/rearrange_pose/'.format(test_name)
    camera_list_path = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/'.format(test_name, pose_index)
    silouettee_image_path = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/'.format(test_name, pose_index)
    estimated_location_file = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/estimated_location.txt'.format(test_name, pose_index)
    original_image_path = "G:/PhDProject_real_data/{}/".format(test_name)
    current_pose = pose_index
    # if use_previous == True
    # load the previous pose matrix as a starting point for the current pose reconstruction
    dataset = image_dataset(camera_meta_path, camera_list_path, silouettee_image_path,original_image_path, current_pose)
    
    batch_size = dataset.camera_number
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for sample in train_dataloader: 
        original_image = sample['original']
        reconstruction = sample['reconstruction']
      

    model = Model(args.template_mesh, estimated_location_file).cuda()
    # get the VGG16 model ready
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg_model = VGG_pretrain() # get the pretrained VGG model
    

    embed_size = 128
    attention_model = Encoder(4096, embed_size, 1, 8, 1, device, 1, 0).to(device)
    for training_sample in train_dataloader: 
        original_image = training_sample['original'].float().cuda()
    
    vgg_image_feature = vgg_model(original_image)
    vgg_image_feature = vgg_image_feature.unsqueeze(0)

    optimizer = torch.optim.Adam(model.parameters(), 0.01,betas=(0.5, 0.99))

    #renderer.transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

    epoch = tqdm.tqdm(list(range(0,epoch)))
    gif_images = []
    writer = imageio.get_writer(os.path.join('./', 'deform_bat_{}.gif'.format(pose_index)), mode='I')
    for i in epoch:
        
        for training_sample in train_dataloader:
      
            images_gt = training_sample['mask'].cuda()
            camera_matrix = training_sample['camera_matrix'].cuda()
            original_image = training_sample['original'].float().cuda()
            reconstruction  =training_sample['reconstruction'].cuda()
            
            #images_gt = torch.from_numpy(images).cuda()
            # at the begining, train the model orientation first
            
            
            # passing the image into the pretrained VGG model for feature extraction
            
           
            # then feed the output into a transformer to get output score, using softmax to normalize it 
                        
            mesh, current_pose, laplacian_loss, attention_score = model(batch_size,vgg_image_feature)
            print(attention_score)
            attention_score = attention_score.squeeze()
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
            IOU_loss = neg_iou_loss_weighted(images_pred[:, -1], images_gt[:, 0], attention_score)

            vertices_loss = torch.norm(mesh.vertices - reconstruction) / batch_size

            

            loss = IOU_loss + vertices_loss
            

           
            loss = IOU_loss 
            epoch.set_description('IOU Loss: %.4f  Vertices Loss: %.4f' % (IOU_loss.item(), vertices_loss.item()))
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        if i % 1 == 0:
            #print("pred_image shape: ",images_pred.detach().cpu().numpy()[0].shape )
            image = images_pred.detach().cpu().numpy()[0].transpose((1 , 2, 0))
            writer.append_data((255*image[:, :, 0]).astype(np.uint8))
            
            for counter in range(1):
                
                image = images_pred.detach().cpu().numpy()[0].transpose((1 , 2, 0))
                imageio.imsave(os.path.join(args.output_dir, 'pred_camera_{}_{}.png'.format(i, counter)), (255*image[..., 1]).astype(np.uint8))
            image_gt = images_gt.detach().cpu().numpy()[0].transpose((1, 2, 0))
            

    model(1,vgg_image_feature)[0].save_obj(os.path.join(args.output_dir, '{}_bat_weight_{}.obj'.format(test_name, pose_index)), save_texture=False)


if __name__ == '__main__':
    start_pose = 1
    end_pose = 1
    interval = 1
    test_name = "brunei_2023_bat_test_13_2"
    for pose_index in range(start_pose, end_pose + interval, interval):
        print("working on pose: ", pose_index)
        epoch = 200
        # developing the use_previous to provide extra supervision
        main(test_name, pose_index, epoch, use_previous = False)
    
    
    