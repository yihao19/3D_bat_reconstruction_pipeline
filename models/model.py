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
import soft_renderer.functional as srf
import cv2 as cv
import math
from collections import Counter
from torch import sin, cos
from torch.autograd import Variable
from torch.utils.data import DataLoader
from chamferdist import ChamferDistance
from LBS import LBS
from torch.nn import functional as F
'''
trying to make the model learning euler angle and displacement
by using camera matrix

'''
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, './')

'''
define the dataset, it contains the sequence of silhouette images (N, H, W, 1)   
                                the sequence of forward optical flow images (N-1, H, W, 2)
                                the sequence of backeward optical flow images (N-1, H, W, 2)
                                the sequence of rough estimated location (N, 3)
                                the sequence of corresponding camera matrix (N, NC, 3, 4)
                                the sequence of camera index that has complete or partial bat (N, string) 
and the camera matrix

camera_list: passing the camera index to control the number of passed camera
and angles: only read the camera meta that the silouette contains part of the bat

'''
class image_dataset(Dataset):
    def __init__(self, camera_meta_path, pose_root_dir, pose_start_index, pose_end_index, image_size, if_optical_info = False):
        self.camera_meta_path = camera_meta_path
        #self.camera_list_path = camera_list_path
        self.pose_start_index = pose_start_index
        self.pose_end_index = pose_end_index
        self.pose_root_dir = pose_root_dir
        self.pose_number = self.pose_end_index - self.pose_start_index + 1
        self.image_size = image_size
        self.if_optical_info = if_optical_info
        #self.silouette_image_path = silouette_image_path
        # read the file
        self.entire_camera_matrix = np.loadtxt(os.path.join(self.camera_meta_path, 'camera_meta.txt'))
        # calculate the optical flow in the init process
        self.mask_image_seq = []
        self.fw_flow_seq = []
        self.bw_flow_seq = []
        self.estimated_location_seq = []
        self.camera_matrix_seq = []
        self.camera_number_list = []
        self.total_camera_seq = [] 
        self.common_camera_index = [] # store the camera index that is common for the poses, 
                                      # for the optical flow loss, only calculated with the common camera
        self.point_cloud_list = [] # store the point cloud estimation of the current pose 
        # using the chamfer loss as another 3D supervision for the model
        self.camera_index_list = []
        for index in range(self.pose_start_index, self.pose_end_index + 1):
            # get the sequence of the silouette images
            mask_path = os.path.join(self.pose_root_dir, "pose{}/masks".format(index))
            camera_list_file = os.path.join(mask_path, "camera.txt")
            camera_list= []
            camera_list_file = open(camera_list_file)
            camera_list_string = camera_list_file.read()
            camera_list_string = camera_list_string[1: len(camera_list_string)-1]
            self.camera_index_list.append(camera_list_string)
            camera_list = camera_list_string.split(', ') 
            for item in camera_list: 
                self.total_camera_seq.append(int(item))
            camera_matrix = np.zeros((len(camera_list), 12))
            self.camera_number_list.append(len(camera_list))
            image_height = self.image_size[0]
            image_width = self.image_size[1]
            counter = 0
            mask_images = np.zeros((len(camera_list), 1, image_height, image_width))
            for index in camera_list: 
                index = int(index)
                image_name = "camera{}.png".format(index)
                image_path = os.path.join(mask_path, image_name)
                mask_image = cv.imread(image_path)[:, :, 0].astype('float32') / 255.
                mask_image = np.expand_dims(mask_image, axis=-1)
                mask_image = mask_image.transpose((2, 0, 1))
                mask_images[counter] = mask_image
                camera_matrix[counter] = self.entire_camera_matrix[index-1]
                counter += 1
            estimated_location_file = open(os.path.join(mask_path, "estimated_location.txt"))
            estimated_location_string = estimated_location_file.read()
            parts = estimated_location_string.split(' ')
            x_average = float(parts[0])
            y_average = float(parts[1])
            z_average = float(parts[2])
            camera_matrix = np.reshape(camera_matrix, (len(camera_list), 3, 4)).astype('float32')
            self.mask_image_seq.append(mask_images)
            self.camera_matrix_seq.append(camera_matrix)
            self.estimated_location_seq.append(np.array([x_average, y_average, z_average]))
            
            
            
            # load the estimated point cloud
            
            estimated_pointcloud_file_name = os.path.join(mask_path, "estimated_point_cloud.txt")
            estimated_point_cloud = np.loadtxt(estimated_pointcloud_file_name)
            self.point_cloud_list.append(estimated_point_cloud)
        # store the camera index that has the same number of duplicates as the pose number
        
        count = Counter(self.total_camera_seq) # check how many cameras are shared by the entire pose sequence
        

      
        for value in count: 
            if(count[value] == self.pose_number):
                self.common_camera_index.append(value)
        
        if(self.if_optical_info == True):
        # calculating the forward and backward optical flow
        # make sure to save the common_camera_index 
            for camera_index in self.common_camera_index: 
                fw_flow = self.forward_optical_flow(camera_index)
                bw_flow = self.backward_optical_flow(camera_index)
                # then use the silouette mask to preprocess the optical flow
                
                self.fw_flow_seq.append(fw_flow)
                #self.bw_flow_seq.append(bw_flow)
  
    def __len__(self):
        # since it is a whole sequence
        return 1
    '''
    function for calculating the forward optical flow and return as a numpy array (N - 1, image_size, 2)
    input: camera_index and silouette_mask
            silouette_mask is for preprocessing the optical flow 
            silouette_mask shape: [pose_number, camera_number, 1, H, W]
    
    def forward_optical_flow(self, camera_index):
        image_height = self.image_size[0]
        image_width = self.image_size[1]
        fw_optical_flow = np.zeros((self.pose_number-1, image_height, image_width, 2))
        
        for pose_index in range(self.pose_start_index, self.pose_end_index):
            prev_frame_path = os.path.join(self.pose_root_dir, "pose{}/camera{}.png".format(pose_index, camera_index))
            next_frame_path = os.path.join(self.pose_root_dir, "pose{}/camera{}.png".format(pose_index + 1, camera_index))
            # using the silouette images to preprocess the flow
            
            prev_silouette_path = os.path.join(self.pose_root_dir, "pose{}/masks/camera{}.png".format(pose_index, camera_index))
            
            
            prev_frame = cv.imread(prev_frame_path)
            next_frame = cv.imread(next_frame_path)
            prev_silouette = cv.imread(prev_silouette_path)[:, :, 0:2].astype('float32') / 255. # forground is 1 and background is 0
            
            # calculate the forward optical flow
            prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
            next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, 
                                               None,
                                               0.5, 3,15, 3, 5, 1.2, 0)
            # flow represent the pixel displacement from prev frame to the next frame
            
            preprocessed_flow = np.multiply(flow, prev_silouette)
            fw_optical_flow[pose_index - self.pose_start_index] = preprocessed_flow
        return fw_optical_flow
    '''
    '''
    function for calculating the backward optical flow and return as a numpy array (N - 1, image_size, 2)
    '''
    '''
    def backward_optical_flow(self,camera_index):
        image_height = self.image_size[0]
        image_width = self.image_size[1]
        bw_optical_flow = np.zeros((self.pose_number - 1, image_height, image_width, 2))
        for pose_index in range(self.pose_end_index, self.pose_start_index, -1):
            prev_frame_path = os.path.join(self.pose_root_dir, "pose{}/camera{}.png".format(pose_index, camera_index))
            next_frame_path = os.path.join(self.pose_root_dir, "pose{}/camera{}.png".format(pose_index - 1, camera_index))
            prev_frame = cv.imread(prev_frame_path)
            next_frame = cv.imread(next_frame_path)
            # calculate the forward optical flow
            prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
            next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, 
                                               None,
                                               0.5, 3,5, 3, 5, 1.2, 0) 
            bw_optical_flow[pose_index - self.pose_start_index-1] = flow
        return bw_optical_flow
    '''
    def __getitem__(self, idx):
       # return the data sample indicated by the passed index
       # swap the 0, 1 index of the optical flow
       
       sample = {'mask': self.mask_image_seq, 
                 'camera_matrix':self.camera_matrix_seq,
                 'camera_string_list':self.camera_index_list, 
                 'point_cloud_list':self.point_cloud_list, 
                 'camera_number': self.camera_number_list, 
                 'estimated_location':self.estimated_location_seq, 
                 'common_camera':self.common_camera_index, 
                 'fw_flow':self.fw_flow_seq, 
                 'bw_flow':self.bw_flow_seq}
       return sample
    
'''
param: template_obj_path: path for template of model in rest pose(obj file)
       bone_skining_matrix_path: path for self-designed bone and default skining_matrix
       joint_list: determine which bone's rotation matrix that you want to trained to get
       train_skining: determine whether you want to train the skining matrix or using 
                      default matrix as hyper-params
'''
class Model(nn.Module):
    def __init__(self, template_obj_path, pose_number, estimated_location,bone_skining_matrix_path='./tunnel_params.pkl', train_skining_matrix = False):
        super(Model, self).__init__()

        # set template mesh
        # the mesh object no need to change since the vertices will move with the
        # joints
        # put the mesh of the model in rest pose in OBJ file and bone and default skining
        # matrix in the corresponding pkl file
        self.estimated_location = estimated_location
        self.pose_number = pose_number
        self.template_mesh = sr.Mesh.from_obj(template_obj_path, load_texture=True, texture_res = 5, texture_type='surface')
        with open(bone_skining_matrix_path, 'rb') as f:
            data = pickle.load(f)
        # generate the .obj file from params.pkl

        self.template_mesh.vertices = torch.tensor(data['v_template'],dtype=torch.float32).unsqueeze(0).cuda()
        
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
        _location = np.zeros((self.pose_number, 3))
        for counter in range(len(self.estimated_location)):
            _location[counter] = self.estimated_location[counter]
        self.random_dis = torch.tensor(_location, dtype=torch.float32) # randomly assign offset for
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
        
        
        
        # trainable parameters
        # small displacement for each pose after the estimated location
        # every pose could be different
        self.register_parameter('displacement', nn.Parameter(torch.zeros(self.pose_number,1,3)))
        #self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))
        # optimize the euler angles pitch, yaw, and roll
        # all in radient
        # default displacement template to the real locaation
         
        # vertices add to the place where the bat is roughly at

        # joints also need to add a displacement where the bat is roughly at
        # if we are dealing with the sequence of a flying
        # we assume that the variance of the global rotaion should be the same 
        # before 
        
        self.register_parameter('pitch', nn.Parameter(torch.zeros(self.pose_number, 1)))
        self.register_parameter('yaw', nn.Parameter(torch.zeros(self.pose_number, 1)))
        self.register_parameter('roll', nn.Parameter(torch.zeros(self.pose_number, 1)))
        
        
        self.register_parameter('joint_0',nn.Parameter(torch.zeros(self.pose_number, 3)))
        
        self.register_parameter('joint_2',nn.Parameter(torch.zeros(self.pose_number, 3))) # 3 DOF
        self.register_parameter('joint_3',nn.Parameter(torch.zeros(self.pose_number, 3)))    # 1 DOF Z
        self.register_parameter('joint_4',nn.Parameter(torch.zeros(self.pose_number, 3))) # 3 DOF
        self.register_parameter('joint_5',nn.Parameter(torch.zeros(self.pose_number, 3)))    # 1 DOF
        self.register_parameter('joint_6',nn.Parameter(torch.zeros(self.pose_number, 3))) # 3 DOF
        self.register_parameter('joint_7',nn.Parameter(torch.zeros(self.pose_number, 3))) # 3 DOF
        self.register_parameter('joint_8',nn.Parameter(torch.zeros(self.pose_number, 3)))    # 1 DOF Z
        self.register_parameter('joint_9',nn.Parameter(torch.zeros(self.pose_number, 3)))    # 1 DOF Z
        self.register_parameter('joint_10',nn.Parameter(torch.zeros(self.pose_number, 3)))# 3 DOF
        self.register_parameter('joint_11',nn.Parameter(torch.zeros(self.pose_number, 3)))   # 1 DOF Z
        self.register_parameter('joint_12',nn.Parameter(torch.zeros(self.pose_number, 3)))# 3 DOF
        self.register_parameter('joint_13',nn.Parameter(torch.zeros(self.pose_number, 3)))   # 1 DOF Z
        self.register_parameter('joint_14',nn.Parameter(torch.zeros(self.pose_number, 3)))# 3 DOF
        self.register_parameter('joint_15',nn.Parameter(torch.zeros(self.pose_number, 3)))   # 1 DOF Z
        self.register_parameter('joint_16',nn.Parameter(torch.zeros(self.pose_number, 3)))# 3 DOF
        self.register_parameter('joint_17',nn.Parameter(torch.zeros(self.pose_number, 3)))   # 1 DOF Z
        
        
        #self.register_parameter("pose_tensor", nn.Parameter(torch.zeros(1,19,3)))
        # learning the joint local transformation for all the pose
        self.pose_tensor = torch.zeros((self.pose_number, 23, 3)).cuda()
        
        #self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        #self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
        #
    
    '''
    # model's forward function'
    # camera_number_list: list same length of pose to record the number of camera with pose
    '''
    def forward(self, camera_number_list):
        
        # using sigmoid / tanh function to limit the rotation degree
        # limit the bone rotation to 45 degree maximum pi/4
        # sigmoid x - > (0, 1) 
        # tanh    x - > (-1, 1)
        # assign all bones with fill DOF but a limited rotation angle
        # construct the local transformation matrix

        for counter in range(self.pose_number):
            # keep adding limit to the degree of freedom of each join along the axis
            self.pose_tensor[counter][0][:] = math.pi / 4 * torch.tanh(self.joint_0[counter])
            #self.pose_tensor[counter][0][:] = self.joint_0[counter]
            
            # shoulder bones 
            # x: [-45, 45]
            # y: [-45, 45]
            # z: [-45, 45]
            self.pose_tensor[counter][3] = math.pi / 2 * torch.tanh(self.joint_3[counter])
            self.pose_tensor[counter][11] = math.pi / 2 * torch.tanh(self.joint_11[counter])
            # arm bones: 
            # x: []
            # y: []
            # z: []
            self.pose_tensor[counter][4][:] = math.pi / 2 * torch.tanh(self.joint_4[counter])
            self.pose_tensor[counter][12][:] = math.pi / 2 * torch.tanh(self.joint_12[counter])
            
            self.pose_tensor[counter][13][:] = math.pi / 4 * torch.tanh(self.joint_13[counter])
            self.pose_tensor[counter][15][:] = math.pi / 4 * torch.tanh(self.joint_15[counter])
            self.pose_tensor[counter][17][:] = math.pi / 4 * torch.tanh(self.joint_17[counter])
            
            self.pose_tensor[counter][5][:] = math.pi / 4 * torch.tanh(self.joint_5[counter])
            self.pose_tensor[counter][7][:] = math.pi / 4 * torch.tanh(self.joint_7[counter])
            self.pose_tensor[counter][9][:] = math.pi / 4 * torch.tanh(self.joint_9[counter])
            '''
            '''
            
              
        '''
        self.pose_tensor[14][:] = math.pi / 6 * torch.tanh(self.joint_14)
        self.pose_tensor[10][:] = math.pi / 6 * torch.tanh(self.joint_10)
        self.pose_tensor[6][:] = math.pi / 6 * torch.tanh(self.joint_6)
        self.pose_tensor[7][:] = math.pi / 6 * torch.tanh(self.joint_7)
        self.pose_tensor[12][:] = math.pi / 6 * torch.tanh(self.joint_12)
        self.pose_tensor[16][:] = math.pi / 6 * torch.tanh(self.joint_16)
        
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
        # render for each pose
        mesh_sequence = []
        joint_sequence = [] #for testing the joint position
        for counter in range(self.pose_number):
            
            
            # deform the template for each pose using the corresponding pose_tensor
            # using the self.vertices, and self.joints since it is all the same
            #vertices, joints = self.LBS_model(self.vertices,self.joints, self.pose_tensor[counter], to_rotmats=True)
            #self.pose_tensor = self.pose_tensor.squeeze()
            '''
            tensor_1 = torch.ones(1).cuda()
            tensor_0 = torch.zeros(1).cuda()
     
            RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, cos(self.roll[counter]), -sin(self.roll[counter])]),
                    torch.stack([tensor_0, sin(self.roll[counter]), cos(self.roll[counter])])]).reshape(3,3).cuda()

            RY = torch.stack([
                            torch.stack([cos(self.pitch[counter]), tensor_0, sin(self.pitch[counter])]),
                            torch.stack([tensor_0, tensor_1, tensor_0]),
                            torch.stack([-sin(self.pitch[counter]), tensor_0, cos(self.pitch[counter])])]).reshape(3,3).cuda()
            
            RZ = torch.stack([
                            torch.stack([cos(self.yaw[counter]), -sin(self.yaw[counter]), tensor_0]),
                            torch.stack([sin(self.yaw[counter]), cos(self.yaw[counter]), tensor_0]),
                            torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3).cuda()
            
            
            rotation =torch.mm(RX, torch.mm(RY, RZ))
            rotation = torch.unsqueeze(rotation, dim=0)
            
            vertices = torch.bmm(self.vertices,rotation) 
            joints = torch.bmm(self.joints,rotation.double())
            '''
            vertices, joints = self.LBS_model(self.vertices,self.joints, self.pose_tensor[counter], to_rotmats=True)
            
            # the maximum displacement is about 3 meters
            displacement = 3 * torch.tanh(self.displacement)
            vertices = vertices + self.random_dis[counter].repeat(1, self.vertices_number, 1).cuda() + displacement[counter].repeat(1, self.vertices_number, 1).cuda() 
            joints = joints + self.random_dis[counter].repeat(1, self.joint_number, 1).cuda() + displacement[counter].repeat(1, self.joint_number, 1).cuda() 
        
            current_mesh = sr.Mesh(vertices.repeat(camera_number_list[counter], 1, 1),self.faces.repeat(camera_number_list[counter], 1, 1))
            mesh_sequence.append(current_mesh)
            joint_sequence.append(joints)
        #self.vertices = vertices #+ self.random_dis.repeat(1, self.vertices_number, 1).cuda()
        
        #self.joints = joints #+ self.random_dis.repeat(1, self.joint_number, 1).cuda()
        '''
        _joints = joints.detach().squeeze().cpu().numpy()
        
        _vertices = vertices.detach().squeeze().cpu().numpy()
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1,projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.azim = -90
        ax.roll = 0
        ax.elev = 90
        ax.axes.set_xlim3d(left=27, right=30)
        ax.axes.set_ylim3d(bottom=-3, top=3) 
        ax.axes.set_zlim3d(bottom=-3, top=3) 
        jx = _vertices[:, 0]
        jy = _vertices[:, 1]
        jz = _vertices[:, 2]
        
        
        ax.scatter(jx,jy,jz,color='b') 
        
        jx = _joints[:, 0]
        jy = _joints[:, 1]
        jz = _joints[:, 2]
        
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.azim = -90
        ax.roll = 0
        ax.elev = 90
        table = self.kintree_table.cpu().numpy()
        #print(table)
        for index in range(self.joint_number):
            if(table[0][index] == -1):
                continue
            else:
                parent_index = table[0][index]
                child_index = table[1][index]
                parent = _joints[parent_index]
                child = _joints[child_index]
                x, y, z = [parent[0], child[0]], [parent[1], child[1]], [parent[2], child[2]]
                ax.text(_joints[child_index][0],_joints[child_index][1],_joints[child_index][2], str(index)) 
                ax.plot(x, y, z, color='black')
        ax.axes.set_xlim3d(left=27, right=30)
        ax.axes.set_ylim3d(bottom=-3, top=3) 
        ax.axes.set_zlim3d(bottom=-3, top=3) 
        ax.scatter(jx,jy,jz,color='r') 
        
        '''
        #return
        #np.savetxt('./test_2.txt', verts.detach().cpu().numpy()[0])
        
        # apply Laplacian and flatten geometry constraints
        #laplacian_loss = self.laplacian_loss(vertices).mean()
        #flatten_loss = self.flatten_loss(vertices).mean()
        # add l2 regularization for small wing bones
        # sequence_rotation_loss: make sure the global rotation of the template is roughly the same
        # we use the variance of the euler angle in the sequence to measure this
        rotation_variance = torch.var(self.joint_0, dim = 0)
        rotation_variance_loss = torch.norm(rotation_variance)
        # symmetric regularization
        
        
        bone_3_11_reg = torch.norm(self.joint_3 + self.joint_11)
        bone_4_12_reg = torch.norm(self.joint_4 + self.joint_12)
        
        
        
        
        bone_reg =  bone_3_11_reg + bone_4_12_reg
        return mesh_sequence, joint_sequence, torch.tensor(0)#0.001 * bone_reg

class Vertices_Projection(nn.Module):
    def __init__(self, batch_number, P=None, dist_coeffs=None, image_size = (1024, 1280)):
        super(Vertices_Projection, self).__init__()
        
        self.P = P
        self.dist_coeffs = dist_coeffs
        self.image_size = image_size
        self.batch_number = batch_number
        '''
        if isinstance(self.P, np.ndarray):
            self.P = torch.from_numpy(self.P).cuda()
        if self.P is None or self.P.ndimension() != 3 or self.P.shape[1] != 3 or self.P.shape[2] != 4:
            raise ValueError('You need to provide a valid (batch_size)x3x4 projection matrix')
        '''
        if dist_coeffs is None:
            self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(self.batch_number, 1)

    def forward(self, vertices, P):
        vertices = self.projection(vertices, P)
        return vertices
    '''
    function that will project the vertices to image and 
    setup the out-of-boundary coordinates to zeros
    '''
    def projection(self, vertices, camera_matrix):
        vertices = torch.cat([vertices, torch.ones_like(vertices[:, :, None, 0])], dim=-1)
        #print("prev_vertices.shape: ", prev_vertices.shape)
        proj_vertices = torch.bmm(vertices, camera_matrix.transpose(2, 1))
        _x, _y, _z = proj_vertices[:, :, 0], proj_vertices[:, :, 1], proj_vertices[:, :, 2]
        _x_coord = (_x / (_z + 1e-6))
        _y_coord = (_y / (_z + 1e-6))
        _x_coord = _x_coord.unsqueeze(-1)
        _y_coord = _y_coord.unsqueeze(-1)
        x_mask_bottom = torch.where(_x_coord[:] >= 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()) 
        x_mask_up = torch.where(_x_coord[:] < self.image_size[1], torch.tensor(1).cuda(), torch.tensor(0).cuda())
        y_mask_bottom = torch.where(_y_coord[:] >= 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()) 
        y_mask_up = torch.where(_y_coord[:] < self.image_size[0], torch.tensor(1).cuda(), torch.tensor(0).cuda())
        x_mask = x_mask_up * x_mask_bottom
        y_mask = y_mask_up * y_mask_bottom
        mask = x_mask * y_mask
        _x_coord = _x_coord * mask
        _y_coord = _y_coord * mask
        _xy_coord = torch.cat([_x_coord, _y_coord], dim = 2)
        return _xy_coord

'''
IOU loss define the
'''                  
def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()



'''
define the chamfer distance
shape of the estimated_vertices: (N, # vertices, 3) 
    shape of the pred_vertices:  (N, # vertices, 3)
'''
def Chamfer_distance(pred_vertices, estimated_vertices):
    pred_vertices = pred_vertices.float()
    prev_vertices_num = pred_vertices.shape[1]
    estimated_vertices = estimated_vertices.float()
    estimated_vertices_num = estimated_vertices.shape[1]
    chamferDist = ChamferDistance()
    dist_forward = chamferDist(pred_vertices, estimated_vertices)/prev_vertices_num +  chamferDist(estimated_vertices, pred_vertices) / estimated_vertices_num
    return dist_forward

'''
function project the vertices on a mesh and return the batch like x and y coordinates
input shape: mesh_vertices: (batch_number, vertices_number, 3)
camera_matrix: (batch_number, 3, 4)
output shape:
    x_coordinate: (batch_number, vertices_number, 1)
    y_coordinate: (batch_number, vertices_number, 1)
    
    x and y coordinate will be using as index later so make them long tensor type
    assume the device is always CUDA
'''
def projection(mesh_vertices, camera_matrix):
    mesh_vertices = torch.cat([mesh_vertices, torch.ones_like(mesh_vertices[:, :, None, 0])], dim=-1)
    #print("prev_vertices.shape: ", prev_vertices.shape)
    proj_vertices = torch.bmm(mesh_vertices, camera_matrix.transpose(2, 1))
    _x, _y, _z = proj_vertices[:, :, 0], proj_vertices[:, :, 1], proj_vertices[:, :, 2]
    _x_coord = ((_x / (_z + 1e-6)))
    _y_coord = ((_y / (_z + 1e-6)))
    _x_coord = _x_coord.unsqueeze(-1)
    _y_coord = _y_coord.unsqueeze(-1)
    
    #prev_xy_coord = torch.where(mask_bottom, prev_xy_coord, torch.tensor([0,0]).to(torch.int).cuda())
    #result_1 = img_feat[torch.arange(batch_size)[:, None], x.view(batch_size, -1), y.view(batch_size, -1), :]
    _x_coord = _x_coord.to(torch.int).cuda()
    _y_coord = _y_coord.to(torch.int).cuda()
    
    return _x_coord, _y_coord

'''
input: prev_mesh, next_mesh for calculating the projection on image x_move, y_move
output: l2 loss of the calculated optical motion and the queried from flow image 

input: prev_mesh, next_mesh, mesh object
       prev_camera_matrix: (image_number, 3, 4)
       fw_flow_gt:         (image_number , H, W, 2)
       image_size:         (H, W)
       
output: queried optical flow
''' 
def fw_flow_queried(mesh, camera_matrix, fw_flow_gt, image_size):
    batch_size = camera_matrix.shape[0]
    vertices_number = mesh.vertices.shape[1]
    image_height = image_size[0]
    image_width = image_size[1]
    # generate a masks image indicate if the mesh point is on the image frame, 0 indicate not, 1 indicate in
    queried_flow = 0
    #mask = np.zeros((batch_size, image_height, image_width))
    
    # get the previous mesh vertices and project using the prev_camera_matrix 
    # output shape: (batch_number, vertex_number, 2)
    
    _vertices = mesh.vertices
    # get the projection
    x_coord, y_coord = projection(_vertices, camera_matrix)
    x_coord = x_coord.long()
    y_coord = y_coord.long()
    
    # get rid of the coordinate that get out of the boundary
    x_mask_bottom = torch.where(x_coord[:] >= 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()) 
    x_mask_up = torch.where(x_coord[:] < image_size[1], torch.tensor(1).cuda(), torch.tensor(0).cuda())
    
    x_mask = x_mask_bottom * x_mask_up
    
    y_mask_bottom = torch.where(y_coord[:] >= 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()) 
    y_mask_up = torch.where(y_coord[:] < image_size[0], torch.tensor(1).cuda(), torch.tensor(0).cuda())
    
    y_mask = y_mask_bottom * y_mask_up
    # get the mask for both x and y 
    mask = x_mask * y_mask
    x_coord = x_coord * mask
    y_coord = y_coord * mask

    # query the motion value
    queried_flow = fw_flow_gt[torch.arange(batch_size)[:, None], y_coord.view(batch_size, -1), x_coord.view(batch_size, -1), :]

    
    # mask all the point that out of the image and replace the value with zeros)
    mask = mask.repeat(1,1,2)
    

    queried_flow = queried_flow * mask
    # get the displacement from the next_mesh
    
    '''
    validate the code and projection using the first images
    
    1. visualize the optical flow
    2. and the projection on the origianl image
    
    prev_fw_flow_gt = prev_fw_flow_gt.cpu().numpy()
    sample_flow = prev_fw_flow_gt[2]
    print(sample_flow.shape)
    fig, axes = plt.subplots(nrows=1, ncols=3)
    im1 = axes[0].imshow(sample_flow[:, :, 0]) 
    im2 = axes[1].imshow(sample_flow[:, :, 1]) 
    prev_image = np.zeros((image_size[0], image_size[1]))
    prev_x_coord = prev_x_coord.cpu().numpy()[2]
    prev_y_coord = prev_y_coord.cpu().numpy()[2]
    
    for index in range(prev_x_coord.shape[0]):
        x = prev_x_coord[index][0]
        y = prev_y_coord[index][0]
        prev_image[y][x] = 1
    im3 = axes[2].imshow(prev_image)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im1, cax=cbar_ax)
    return 
    '''
    return queried_flow
    
    
def bw_flow_l2_loss(prev_mesh, next_mesh, camera_matrix, bw_flow_gt):
    
    return
    

def main(pose_index):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template-mesh', type=str,
                        default=os.path.join(data_dir, './pose150.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
                        default=os.path.join(data_dir, 'results/output_deform'))
    args = parser.parse_args()
    
    # make the model data loader style of input 
    os.makedirs(args.output_dir, exist_ok=True)


    # start hyper parameters
    image_size = (1024,1280)
    total_epoch = 300
    stage_2_start = 300
    
    if_optical_info = False # setting the training data to determine if the optical flow info is used
                             # or silouette image only
    learning_rate = 0.01
    pose_start_index = pose_index
    pose_end_index = pose_index
    flow_loss_weight = 0.0005
    chamfer_weight = 0.000
    pose_number = pose_end_index - pose_start_index + 1
    #output_path = 'G:\GCN_project/Bat_Sample_Model/rast_bat_images/non_square/rearranged/pose{}/masks/'.format(_pose_index)
    #camera_list = [1]
    
    #camera_list_path = "G:/GCN_project/Bat_Sample_Model/Rearranged_dataset_tunnel_simulation/rearranged/pose{}/masks/".format(_pose_index)
    #silouettee_image_path = "G:/GCN_project/Bat_Sample_Model/Rearranged_dataset_tunnel_simulation/rearranged/pose{}/masks/".format(_pose_index)
    #estimated_location_file = "G:/GCN_project/Bat_Sample_Model/Rearranged_dataset_tunnel_simulation/rearranged/pose{}/masks/estimated_location.txt".format(_pose_index)
    pose_root_dir    = "G:/GCN_project/Bat_Sample_Model/high_speed_tunnel/rearranged/"

    camera_meta_path = "G:/GCN_project/Bat_Sample_Model/high_speed_tunnel/rearranged/"
    dataset = image_dataset(camera_meta_path, pose_root_dir,  pose_start_index, pose_end_index,image_size, if_optical_info)
    
    batch_size = 1 # only one gaint sequence
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for sample in train_dataloader:
        estimated_location_seq = sample['estimated_location']
        images_gt = sample['mask'][0].cuda()
        camera_matrix = sample['camera_matrix'][0].cuda()
        point_cloud = sample['point_cloud_list'][0].cuda()
        
    

    model = Model(args.template_mesh, pose_number, estimated_location_seq).cuda()
    
    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.5, 0.99))

    #renderer.transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

    epoch = tqdm.tqdm(list(range(0,total_epoch)))
    #gif_images = []
    #writer = imageio.get_writer(os.path.join('./', 'deform_bat_{}.gif'.format(_pose_index)), mode='I')
    for i in epoch:
        
        for training_sample in train_dataloader:
            
            #images_gt = training_sample['mask'].cuda()
            #camera_matrix = training_sample['camera_matrix'].cuda()
            #images_gt = torch.from_numpy(images).cuda()
            camera_number = training_sample['camera_number']
            # we get all the reconstructed mesh
            # start re-render the silouette images
            mesh_sequence, joint_sequence, rotation_variance_loss = model(camera_number) # return a sequence of mesh corresponding to the "start_pose" to "end_pose"
            total_loss = 0
            total_iou_loss = 0
            fw_loss = 0
            chamfer_loss = 0
            #total_fw_loss  = 0
            # calculate the IOU loss 
            # get all the input
            camera_matrices = training_sample['camera_matrix']
            images_gts = training_sample['mask']
            camera_string_lists = training_sample['camera_string_list']
            estimated_point_cloud_list = training_sample['point_cloud_list']
            if(if_optical_info == True):
                fw_flow_gt = training_sample['fw_flow']
                fw_flow_gt = torch.cat(fw_flow_gt, dim = 0)
                fw_flow_gt = torch.transpose(fw_flow_gt, 0, 1)

            for pose_index in range(pose_number):
                images_gt = images_gts[pose_index].squeeze(0).cuda()
                camera_matrix = camera_matrices[pose_index].squeeze(0).cuda()
                camera_string_list = camera_string_lists[pose_index]
                estimated_point_cloud = estimated_point_cloud_list[pose_index].cuda()
                #print("camera_string_list: ", camera_string_list)
                renderer = sr.SoftRenderer(image_height=image_size[0], image_width=image_size[1],sigma_val=1e-6,
                                       camera_mode='projection', P = camera_matrix,orig_height=image_size[0], orig_width=image_size[1], 
                                       near=0, far=100)
                
                # check the mesh vertices and the projection
                # for two consecutive poses
                # each pose has 10 images, 
                # image_coordinate on x and y shape: x:[10, 472], y:[10, 472]
                # calculated vertices movement between two consecutive frame
                # x_move: [10, 472], y_move: [10, 472]
                
                # flow gt image: a list indexed by camera
                # gt[camera_index] = [pose_number, 1024, 1280, 2]
                
                # desired output from function: 
                # using the coordinate xy:[10, 472, 2]
                # and gt: [10, 1024, 1280, 2]
                # output shape: flow_x:[10, 472], flow_y:[10, 472]
                
                
                
                
                # calculate the optical loss only when the common camera list is not empty
                # if the common camera list is empty, which means that no camera set are shared 
                # by the sequence of the pose, then only silouette loss is enabled
                if(if_optical_info == True):
                    projection_transformer = Vertices_Projection(8)
                    if(pose_index < pose_number - 1):
                        # reach the last pose whose fw_flow is not available 
                        prev_mesh = mesh_sequence[pose_index]
                        next_mesh = mesh_sequence[pose_index + 1]
                        prev_camera_matrix = camera_matrices[pose_index].squeeze(0).cuda()
                        next_camera_matrix = camera_matrices[pose_index+1].squeeze(0).cuda()
                        prev_fw_flow_gt = fw_flow_gt[pose_index].cuda()
                        
                        
                        '''
                        validate the code and projection using the first images
                        
                        1. visualize the optical flow
                        2. and the projection on the origianl image
                        
                        sample_flow = prev_fw_flow_gt.cpu().numpy()[2]
                        #sample_flow = prev_fw_flow_gt[2]
                        #print(sample_flow.shape)
                        fig, axes = plt.subplots(nrows=1, ncols=2)
                        im1 = axes[0].imshow(sample_flow[:, :, 0]) 
                        im2 = axes[1].imshow(sample_flow[:, :, 1]) 
                        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
                        fig.colorbar(im1, cax=cbar_ax)
                        return 
                        '''
                        # re-design the projection_transformer to make sure that
                        # only in bounds coordinate is kept
                        # make the out-of -bound coordinate 0
                        prev_vertices = projection_transformer(prev_mesh.vertices, prev_camera_matrix)
                        next_vertices = projection_transformer(next_mesh.vertices, next_camera_matrix)
                        #fw_loss += fw_flow_l2_loss(prev_mesh, next_mesh, prev_camera_matrix, prev_fw_flow_gt, image_size)
                        #print("fw_loss: ", fw_loss)
                        #this will be the loss of the pixel move on the image
                        
                        # then get the queried flow 
                        fw_flow = fw_flow_queried(prev_mesh, prev_camera_matrix, prev_fw_flow_gt, image_size)
                        
                        # generate a shape with 100 pixels in both x and  y directions
                        
                        
                        #displacement = torch.tensor([[300, 300]]).cuda()
                        #displacement = displacement.repeat(472, 1).cuda()
                        fw_loss += flow_loss_weight * torch.norm(prev_vertices[:, :, :2] - next_vertices[:, :, :2] - fw_flow) 
                        #print(fw_flow)
                        #print(next_vertices[:, :, :2] - prev_vertices[:, :, :2])
                        
                        #return
                        # then get 
                # calculate the IOU loss
                images_pred = renderer.render_mesh(mesh_sequence[pose_index])
                # checking the reprojection matrix and vertices
                #print("image_pred shape: ", images_pred.shape)
                # optimize mesh with silhouette reprojection error and
                # geometry constraints
                # silhouette image predicted will in the 4th element of the vector 
                IOU_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])
                
                chamfer_dist = Chamfer_distance(mesh_sequence[pose_index].vertices[0].unsqueeze(0), estimated_point_cloud)
                total_iou_loss += IOU_loss #+ 1 * l2_norm 
                chamfer_loss += chamfer_weight * chamfer_dist
            
            #fw_loss = Variable(1 * fw_loss, requires_grad = True)
            
            if(i < stage_2_start):
                total_loss = (total_iou_loss + chamfer_loss) / pose_number 
                #total_loss = (total_iou_loss +  rotation_variance_loss + fw_loss )/pose_number
            else: 
                total_loss = (total_iou_loss + fw_loss )/pose_number
                
            if(if_optical_info == True):
                epoch.set_description('Total Loss: %.4f, IOU Loss: %.4f, Rot Var: %.4f, Optical_loss: %.4f'  % (total_loss.item(),total_iou_loss.item(), rotation_variance_loss.item(),fw_loss.item()))
            else:
                epoch.set_description('Total Loss: %.4f, IOU Loss: %.4f, Rot Var: %.4f, Optical_loss: %.4f'  % (total_loss.item(),total_iou_loss.item(), rotation_variance_loss.item(),fw_loss))
            
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()
        
        if i % 1 == 0:
            #print("pred_image shape: ",images_pred.detach().cpu().numpy()[0].shape )
            image = images_pred.detach().cpu().numpy()[0].transpose((1 , 2, 0))
            #writer.append_data((255*image[:, :, 0]).astype(np.uint8))
            
            for counter in range(1):
                image = images_pred.detach().cpu().numpy()[0].transpose((1 , 2, 0))
                imageio.imsave(os.path.join(args.output_dir, 'pred_camera_{}_{}.png'.format(i, counter)), (255*image[..., -1]).astype(np.uint8))
            image_gt = images_gt.detach().cpu().numpy()[0].transpose((1, 2, 0))
            
           
            
            
            
            #imageio.imsave(os.path.join(args.output_dir, 'deform_gt_%05d.png' % i), (255*image_gt[..., 0]).astype(np.uint8))
    #imageio.mimsave('./bat_deform.gif', gif_images, format='GIF', duration=1)        
    # save optimized mesh
    mesh_sequence, joint_sequence, rotation_loss = model(camera_number)
    for output_counter in range(pose_number):
        
        mesh_sequence[output_counter].save_obj(os.path.join(args.output_dir, 'chamfer/seq_bat_{}.obj'.format(pose_start_index + output_counter)), save_texture=False)
    

if __name__ == '__main__':
    
    
    start_index = 162
    end_index = 800 + 1
    for pose_index in range(start_index, end_index):
        print("Working on pose: ", pose_index)
        main(pose_index)
    
    
    