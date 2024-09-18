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
            estimated_location_file = open(os.path.join(self.silouette_image_path, 'estimated_location.txt'))
            estimated_location_string = estimated_location_file.read()
            parts = estimated_location_string.split(' ')
            x_average = float(parts[0])
            y_average = float(parts[1])
            z_average = float(parts[2])
            estimated_location = np.array([x_average, y_average, z_average]).astype('float32') # randomly assign offset for

        else:
            prev_pose_index = self.current_pose - 1
            current_path = Path(self.silouette_image_path)
            curr_root = current_path.parent
            prev_output_path = os.path.join(curr_root, str(prev_pose_index), "output.json")
            file = open(prev_output_path)
            prev_output = json.load(file) 
            prev_pose = np.array(prev_output['pose'])
            estimated_location = np.array(prev_output['template_displacement']).astype('float32')
        sample = {'mask': mask_image, 
                  'camera_matrix':camera_matrix.astype('float32'), 
                  'prev_pose':prev_pose, 
                  'estimated_location':estimated_location}
        return sample
    
'''
param: template_obj_path: path for template of model in rest pose(obj file)
       bone_skining_matrix_path: path for self-designed bone and default skining_matrix
       joint_list: determine which bone's rotation matrix that you want to trained to get
       train_skining: determine whether you want to train the skining matrix or using 
                      default matrix as hyper-params
'''
class Model(nn.Module):
    def __init__(self, template_obj_path,bone_skining_matrix_path='./new_bat_params_version2_forward_membrane_double_vertex.pkl', train_skining_matrix = False):
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
            self.register_buffer("skining_adjust",torch.zeros_like(skining_tensor))
            print("shape of skining_tensor: ",skining_tensor.shape)
        else: 
            # make skinging_tensor a trainable parameters
            # make the skining matrix trainable just like the joint rotation
            skining_tensor  = torch.tensor(skining).unsqueeze(0).cuda()
            self.register_buffer('skining',skining_tensor)
            self.register_parameter("skining_adjust", nn.Parameter(torch.zeros_like(skining_tensor)))
            #self.skining = skining_tensor
        # importing the bones and skining matrix of a bat model
        # make the skining matrix the registered param
        # and joints the registered param
        # first, test the 
        joints_tensor = torch.tensor(joints).unsqueeze(0).cuda()
        # define the kintree of the skeleton of the bat
        # define in the Blender 
        kintree_table = np.array([[ -1, 0, 1, 2, 2, 4, 5, 6, 7, 8, 9, 7, 11, 12, 13, 14, 15, 16, 7, 18, 19, 20, 21, 22, 23, 7, 25, 26, 7, 28, 29, 30, 31, 32, 33, 7, 35, 36, 37, 38, 39, 40, 7, 42, 43, 44, 45, 46, 47, 48, 7, 50, 51, 52, 53, 54, 55, 56, 7, 58, 59, 60, 61, 62, 63, 64, 7, 66, 67, 6, 69, 70, 71, 72, 73, 74, 75, 76, 6, 78, 79, 80, 81, 82, 83, 84, 85, 6, 87, 88, 89, 90, 91, 92, 93, 94, 5, 96, 97, 98, 99, 100, 101, 102, 103, 5, 105, 106, 107, 108, 109, 110, 111, 112, 5, 114, 115, 116, 117, 118, 119, 120, 121, 5, 123, 124, 125, 126, 127, 128, 129, 130, 5, 132, 133, 134, 135, 136, 137, 138, 139, 140, 2, 142, 143, 144, 145, 146, 147, 145, 149, 150, 151, 152, 153, 154, 155, 145, 157, 158, 159, 160, 161, 162, 163, 145, 165, 166, 145, 168, 169, 170, 171, 172, 173, 174, 145, 176, 177, 178, 179, 180, 181, 182, 145, 184, 185, 186, 187, 188, 189, 190, 191, 145, 193, 194, 195, 196, 197, 198, 199, 200, 145, 202, 203, 204, 205, 206, 207, 208, 209, 145, 211, 212, 144, 214, 215, 216, 217, 218, 219, 220, 221, 144, 223, 224, 225, 226, 227, 228, 229, 230, 144, 232, 233, 234, 235, 236, 237, 238, 239, 143, 241, 242, 243, 244, 245, 246, 247, 248, 249, 143, 251, 252, 253, 254, 255, 256, 257, 258, 259, 143, 261, 262, 263, 264, 265, 266, 267, 268, 269, 143, 271, 272, 273, 274, 275, 276, 277, 278, 143, 280, 281, 282, 283, 284, 285, 286, 287, 0, 289, 0, 291],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292]])
        # define the pose matrix for the joints in the passing list
 
        # empty pose for all bones
        
         
    
        #cuda all the parameters
        #trainable parameters 
        self.kintree_table = torch.tensor(kintree_table).cuda()
        self.register_buffer('joints',joints_tensor)
        self.register_buffer('vertices', self.template_mesh.vertices)
        self.register_buffer('faces', self.template_mesh.faces)
        self.parents = self.kintree_table[0].type(torch.LongTensor)
        
        # for each bone, then 
        self.training_skining_weight = torch.clone(self.skining)
        if(train_skining_matrix == True):
            weights_mask =  torch.tensor(data['weights_mask']).unsqueeze(0).cuda()
            
            training_skining_weight_scope = (self.skining + self.skining_adjust) * weights_mask
            # replace all zero value to -inf
            self.training_skining_weight =training_skining_weight_scope - torch.where(training_skining_weight_scope != 0, torch.zeros_like(training_skining_weight_scope), torch.ones_like(training_skining_weight_scope) * float('inf'))
            self.training_skining_weight = torch.softmax(self.training_skining_weight, dim = 2)
        
        
    
        
        self.LBS_model = LBS(self.joints, self.parents, self.training_skining_weight)# define the LBS model
        self.vertices_number = self.template_mesh.num_vertices
        # optimize for displacement of the center of the mesh ball
        self.register_parameter('displacement', nn.Parameter(torch.zeros(1,1,3)))
        self.register_parameter('scale', nn.Parameter(torch.ones(1)))
        self.register_parameter('local_adjust', nn.Parameter(torch.zeros(1, self.vertices_number, 3))) # apply a small local adjustment on the template 
                                                                                                       # to adjust the template
        
        #self.register_parameter('pitch', nn.Parameter(torch.zeros(1)))
        #self.register_parameter('yaw', nn.Parameter(torch.zeros(1)))
        #self.register_parameter('roll', nn.Parameter(torch.zeros(1)))
        
        
        # skeleton: 
        self.register_parameter('joint_0',nn.Parameter(torch.zeros(1, 3))) # body overall
        self.register_parameter('joint_1',nn.Parameter(torch.zeros(1, 3))) # body section 1
        self.register_parameter('joint_2',nn.Parameter(torch.zeros(1, 3))) # body section 2
        self.register_parameter('joint_3',nn.Parameter(torch.zeros(1, 3))) # head   
        self.register_parameter('joint_4',nn.Parameter(torch.zeros(1, 3))) # right shoulder
        self.register_parameter('joint_5',nn.Parameter(torch.zeros(1, 3))) # right arm  
        self.register_parameter('joint_6',nn.Parameter(torch.zeros(1, 3))) # right fore-arm section 1
        self.register_parameter('joint_7',nn.Parameter(torch.zeros(1, 3))) # right fore-arm section 2
        self.register_parameter('joint_8',nn.Parameter(torch.zeros(1, 3))) # right finger 1 section 1
        self.register_parameter('joint_9',nn.Parameter(torch.zeros(1, 3))) # right finger 1 section 2
        self.register_parameter('joint_10',nn.Parameter(torch.zeros(1, 3)))# right finger 1 section 3
        self.register_parameter('joint_25',nn.Parameter(torch.zeros(1, 3)))# right finger 2 section 1
        self.register_parameter('joint_26',nn.Parameter(torch.zeros(1, 3)))# right finger 2 section 2
        self.register_parameter('joint_27',nn.Parameter(torch.zeros(1, 3)))# right finger 2 section 3
        self.register_parameter('joint_66',nn.Parameter(torch.zeros(1, 3)))# right finger 3 section 1
        self.register_parameter('joint_67',nn.Parameter(torch.zeros(1, 3)))# right finger 3 section 2
        self.register_parameter('joint_68',nn.Parameter(torch.zeros(1, 3)))# right finger 3 section 3
        self.register_parameter('joint_142',nn.Parameter(torch.zeros(1, 3))) # left shoulder
        self.register_parameter('joint_143',nn.Parameter(torch.zeros(1, 3))) # left arm  
        self.register_parameter('joint_144',nn.Parameter(torch.zeros(1, 3))) # left fore-arm section 1
        self.register_parameter('joint_145',nn.Parameter(torch.zeros(1, 3))) # left fore-arm section 2
        self.register_parameter('joint_146',nn.Parameter(torch.zeros(1, 3))) # left finger 1 section 1
        self.register_parameter('joint_147',nn.Parameter(torch.zeros(1, 3))) # left finger 1 section 2
        self.register_parameter('joint_148',nn.Parameter(torch.zeros(1, 3)))# left finger 1 section 3
        self.register_parameter('joint_165',nn.Parameter(torch.zeros(1, 3)))# left finger 2 section 1
        self.register_parameter('joint_166',nn.Parameter(torch.zeros(1, 3)))# left finger 2 section 2
        self.register_parameter('joint_167',nn.Parameter(torch.zeros(1, 3)))# left finger 2 section 3
        self.register_parameter('joint_211',nn.Parameter(torch.zeros(1, 3)))# left finger 3 section 1
        self.register_parameter('joint_212',nn.Parameter(torch.zeros(1, 3)))# left finger 3 section 2
        self.register_parameter('joint_213',nn.Parameter(torch.zeros(1, 3)))# left finger 3 section 3
        self.register_parameter('joint_291',nn.Parameter(torch.zeros(1, 3)))# right foot  section 1
        self.register_parameter('joint_292',nn.Parameter(torch.zeros(1, 3)))# right foot section 2
        self.register_parameter('joint_289',nn.Parameter(torch.zeros(1, 3)))# left foot section 1
        self.register_parameter('joint_290',nn.Parameter(torch.zeros(1, 3)))# left foot sectio 2
        
        
        
        
        # right wing membrane section 1
        self.register_parameter('joint_149',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_150',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_151',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_152',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_153',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_154',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_155',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_156',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_157',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_158',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_159',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_160',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_161',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_162',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_163',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_164',nn.Parameter(torch.zeros(1, 3)))
        # right wing membrane section 2
        self.register_parameter('joint_168',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_169',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_170',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_171',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_172',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_173',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_174',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_175',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_176',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_177',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_178',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_179',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_180',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_181',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_182',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_183',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_184',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_185',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_186',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_187',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_188',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_189',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_190',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_191',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_192',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_193',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_194',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_195',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_196',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_197',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_198',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_199',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_200',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_201',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_202',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_203',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_204',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_205',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_206',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_207',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_208',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_209',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_210',nn.Parameter(torch.zeros(1, 3)))
        # right wing membrane section 3
        self.register_parameter('joint_214',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_215',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_216',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_217',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_218',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_219',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_229',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_221',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_222',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_223',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_224',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_225',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_226',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_227',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_228',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_229',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_230',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_231',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_232',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_233',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_234',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_235',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_236',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_237',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_238',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_239',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_240',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_241',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_242',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_243',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_244',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_245',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_246',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_247',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_248',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_249',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_250',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_251',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_252',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_253',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_254',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_255',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_256',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_257',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_258',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_259',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_260',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_261',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_262',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_263',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_264',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_265',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_266',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_267',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_268',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_269',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_270',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_271',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_272',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_273',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_274',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_275',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_276',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_277',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_278',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_279',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_280',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_281',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_282',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_283',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_284',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_285',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_286',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_287',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_288',nn.Parameter(torch.zeros(1, 3)))
        
        # left wing membrane  section 1
        self.register_parameter('joint_11',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_12',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_13',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_14',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_15',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_16',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_17',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_18',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_19',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_20',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_21',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_22',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_23',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_24',nn.Parameter(torch.zeros(1, 3)))
        # left wing membrane section 2
        self.register_parameter('joint_28',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_29',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_30',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_31',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_32',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_33',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_34',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_35',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_36',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_37',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_38',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_39',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_40',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_41',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_42',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_43',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_44',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_45',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_46',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_47',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_48',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_49',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_50',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_51',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_52',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_53',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_54',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_55',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_56',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_57',nn.Parameter(torch.zeros(1, 3)))
        
        
        self.register_parameter('joint_58',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_59',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_60',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_61',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_62',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_63',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_64',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_65',nn.Parameter(torch.zeros(1, 3)))
        
        # left wing membrane section 3 
        self.register_parameter('joint_69',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_70',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_71',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_72',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_73',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_74',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_75',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_76',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_77',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_78',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_79',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_80',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_81',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_82',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_83',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_84',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_85',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_86',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_87',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_88',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_89',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_90',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_91',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_92',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_93',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_94',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_95',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_96',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_97',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_98',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_99',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_100',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_101',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_102',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_103',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_104',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_105',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_106',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_107',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_108',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_109',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_110',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_111',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_112',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_113',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_114',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_115',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_116',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_117',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_118',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_119',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_120',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_121',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_122',nn.Parameter(torch.zeros(1, 3)))
       
        
        self.register_parameter('joint_123',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_124',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_125',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_126',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_127',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_128',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_129',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_130',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_131',nn.Parameter(torch.zeros(1, 3)))
        
        self.register_parameter('joint_132',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_133',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_134',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_135',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_136',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_137',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_138',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_139',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_140',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_141',nn.Parameter(torch.zeros(1, 3)))
        

        # make small displacement of the 
        #self.register_parameter("pose_tensor", nn.Parameter(torch.zeros(1,19,3)))
        # add the pose_tensor of the previous mesh model
        self.pose_tensor = torch.zeros((293, 3)).cuda()
        
        self.laplacian_smoothing = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        
        #print(self.laplacian_loss.laplacian)
        #laplacian_loss = self.laplacian_loss(self.vertices).mean()
        #print(laplacian_loss)
        #self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
        
        # start the spring mesh test. 
        # assume the template is where the membrane is in the relax state
        # for testing armature index
        # [68, 77, 86, 95, 104, 113, 122, 131, 141, 292]
        
        # calculate the default distance for each pair
        self.membrane_tip = [68, 77, 86, 95, 104, 113, 122, 131, 141, 292]
        
        
        self.rest_distance = self.spring_rest_distance()
        
    
        
    def spring_rest_distance(self): 
        rest_distance = []
        for counter in range(len(self.membrane_tip)-1): 
            first_joint_index = self.membrane_tip[counter]
            second_joint_index =self.membrane_tip[counter + 1]
            first_joint_tip = self.joints[0][first_joint_index]
            second_joint_tip = self.joints[0][second_joint_index]
            rest_distance.append(torch.norm(second_joint_tip - first_joint_tip) * 0.0035)
            
        return rest_distance
    
    def spring_mesh_loss(self):
        return
    
    '''
    # model's forward function'
    
    '''
    def forward(self, batch_size, estimated_location, use_previous = False, prev_pose = None):
       
        # define how the vertices is going to change
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
        
        if(use_previous == True):
            
            
            self.pose_tensor[0][:] = (prev_pose[0][:] + math.pi / 2 * torch.tanh(self.joint_0))
            
            #self.pose_tensor[1][:] = ( prev_pose[1][:] + math.pi / 18 * torch.tanh(self.joint_1))
            #self.pose_tensor[2][:] = ( prev_pose[2][:] + math.pi / 18 * torch.tanh(self.joint_2))
            #self.pose_tensor[3][:] = ( prev_pose[3][:] + math.pi / 18 * torch.tanh(self.joint_3))
            
            
            #self.pose_tensor[1][:] = math.pi / 4 * torch.tanh(self.joint_1)
            #self.pose_tensor[2][:] = math.pi / 9 * torch.tanh(self.joint_2)  # make the neck bone trainable(slightly)
            # for the shoulder using the symmetric deformation
            self.pose_tensor[5][0] = (prev_pose[5][0] + math.pi / 3 * torch.tanh(self.joint_5[0][0]))
            self.pose_tensor[5][1] = (prev_pose[5][1] + math.pi / 3 * torch.tanh(self.joint_5[0][1]))
            self.pose_tensor[5][2] =  (prev_pose[5][2] + math.pi / 3 * torch.tanh(self.joint_5[0][2]))
            
            self.pose_tensor[18][0] = (prev_pose[18][0] + math.pi / 3 * torch.tanh(self.joint_18[0][0]))
            self.pose_tensor[18][1] =  (prev_pose[18][1] + math.pi / 3 * torch.tanh(self.joint_18[0][1]))
            self.pose_tensor[18][2] =  (prev_pose[18][2] + math.pi / 3 * torch.tanh(self.joint_18[0][2]))
            
            #self.pose_tensor[4][0] =  torch.max(torch.min( prev_pose[4][0] + math.pi / 3 * torch.tanh(self.joint_4[0][0]), upper_bound_2),  lower_bound_2)
            #self.pose_tensor[4][1] =  torch.max(torch.min( prev_pose[4][1] + math.pi / 3 * torch.tanh(self.joint_4[0][1]), upper_bound_2),  lower_bound_2)
            self.pose_tensor[4][2] =  ( prev_pose[4][2] + math.pi / 9 * torch.tanh(self.joint_4[0][2]))
            
            #self.pose_tensor[17][0] =  torch.max(torch.min( prev_pose[17][0] + math.pi / 3 * torch.tanh(self.joint_4[0][0]), upper_bound_2), lower_bound_2)
            #self.pose_tensor[17][1] =  torch.max(torch.min( prev_pose[17][1] - math.pi / 3 * torch.tanh(self.joint_4[0][1]), upper_bound_2), lower_bound_2)
            self.pose_tensor[17][2] = ( prev_pose[17][2] - math.pi / 9 * torch.tanh(self.joint_4[0][2]))
            
            self.pose_tensor[6][0] =  ( prev_pose[6][0]   + math.pi / 6 * torch.tanh(self.joint_6[0][0]))
            self.pose_tensor[6][1] =  ( prev_pose[6][1]   + math.pi / 6 * torch.tanh(self.joint_6[0][1]))
            self.pose_tensor[6][2] =  ( prev_pose[6][2]   + math.pi / 6 * torch.tanh(self.joint_6[0][2]))
            self.pose_tensor[19][0] =  ( prev_pose[19][0] + math.pi / 6 * torch.tanh(self.joint_19[0][0]))
            self.pose_tensor[19][1] =  ( prev_pose[19][1] + math.pi / 6 * torch.tanh(self.joint_19[0][1]))
            self.pose_tensor[19][2] =  ( prev_pose[19][2] + math.pi / 6 * torch.tanh(self.joint_19[0][2]))
            
            #self.pose_tensor[7][:] = ( prev_pose[7][:]   + math.pi / 3 * torch.tanh(self.joint_7))
            #self.pose_tensor[20][:] = ( prev_pose[20][:] + math.pi / 3 * torch.tanh(self.joint_20))
            
            #self.pose_tensor[15][:] = ( prev_pose[15][:] +math.pi / 18 * torch.tanh(self.joint_15))
            #self.pose_tensor[16][:] = ( prev_pose[16][:] +math.pi / 18 * torch.tanh(self.joint_16))
            
            self.pose_tensor[8][:] = ( prev_pose[8][:]   + math.pi / 6 * torch.tanh(self.joint_8))
            #self.pose_tensor[9][:] = ( prev_pose[9][:]   + math.pi / 18 * torch.tanh(self.joint_9))
            #self.pose_tensor[10][:] = ( prev_pose[10][:] + math.pi / 18 * torch.tanh(self.joint_10))
            
            self.pose_tensor[11][:] = ( prev_pose[11][:] + math.pi / 6 * torch.tanh(self.joint_11))
            #self.pose_tensor[12][:] = ( prev_pose[12][:] + math.pi / 18 * torch.tanh(self.joint_12))
            #self.pose_tensor[13][:] = ( prev_pose[13][:] + math.pi / 18 * torch.tanh(self.joint_13))
            
            self.pose_tensor[14][:] = ( prev_pose[14][:] + math.pi / 6 * torch.tanh(self.joint_14))
            #self.pose_tensor[15][:] = ( prev_pose[15][:] + math.pi / 18 * torch.tanh(self.joint_15))
            #self.pose_tensor[16][:] = ( prev_pose[16][:] + math.pi / 18 * torch.tanh(self.joint_16))
            
            self.pose_tensor[21][:] = ( prev_pose[21][:] + math.pi / 6 * torch.tanh(self.joint_21))
            #self.pose_tensor[22][:] = ( prev_pose[22][:] + math.pi / 18 * torch.tanh(self.joint_22))
            #self.pose_tensor[23][:] = ( prev_pose[23][:] + math.pi / 18 * torch.tanh(self.joint_23))
            
            self.pose_tensor[24][:] = ( prev_pose[24][:] + math.pi / 6 * torch.tanh(self.joint_24))
            #self.pose_tensor[25][:] = ( prev_pose[25][:] + math.pi / 18 * torch.tanh(self.joint_25))
            #self.pose_tensor[26][:] = ( prev_pose[26][:] + math.pi / 18 * torch.tanh(self.joint_26))
            
            self.pose_tensor[27][:] = ( prev_pose[27][:] + math.pi / 6 * torch.tanh(self.joint_27))
            #self.pose_tensor[28][:] = ( prev_pose[28][:] + math.pi / 18 * torch.tanh(self.joint_28))
           # self.pose_tensor[29][:] = ( prev_pose[29][:] + math.pi / 18 * torch.tanh(self.joint_29))
            
            self.pose_tensor[30][:] = ( prev_pose[30][:] + math.pi / 6 * torch.tanh(self.joint_30))
            self.pose_tensor[31][:] = ( prev_pose[31][:] + math.pi / 18 * torch.tanh(self.joint_31))
            
            self.pose_tensor[32][:] = ( prev_pose[32][:] + math.pi / 6 * torch.tanh(self.joint_32))
            self.pose_tensor[33][:] = ( prev_pose[33][:] + math.pi / 18 * torch.tanh(self.joint_33))
            '''
            self.pose_tensor[0][:] = math.pi / 2 * torch.tanh(self.joint_0)
            
            self.pose_tensor[1][:] = math.pi / 9 * torch.tanh(self.joint_1)
            self.pose_tensor[2][:] = math.pi / 9 * torch.tanh(self.joint_2)
            self.pose_tensor[3][:] = math.pi / 9 * torch.tanh(self.joint_3)
            
            
            #self.pose_tensor[1][:] = math.pi / 4 * torch.tanh(self.joint_1)
            #self.pose_tensor[2][:] = math.pi / 9 * torch.tanh(self.joint_2)  # make the neck bone trainable(slightly)
            # for the shoulder using the symmetric deformation
            self.pose_tensor[5][0] = math.pi / 3 * torch.tanh(self.joint_5[0][0])
            self.pose_tensor[5][1] = math.pi / 3 * torch.tanh(self.joint_5[0][1])
            self.pose_tensor[5][2] = math.pi / 3 * torch.tanh(self.joint_5[0][2])
            
            
            self.pose_tensor[18][0] = math.pi / 3 * torch.tanh(self.joint_5[0][0])
            self.pose_tensor[18][1] = -math.pi / 3 * torch.tanh(self.joint_5[0][1])
            self.pose_tensor[18][2] = -math.pi / 3 * torch.tanh(self.joint_5[0][2])
            
            self.pose_tensor[6][:] = math.pi / 9 * torch.tanh(self.joint_6)
            self.pose_tensor[19][:] = math.pi / 9 * torch.tanh(self.joint_19)
            
            self.pose_tensor[7][:] = math.pi / 9 * torch.tanh(self.joint_7)
            self.pose_tensor[20][:] = math.pi / 9 * torch.tanh(self.joint_20)
            
            self.pose_tensor[15][:] = math.pi / 18 * torch.tanh(self.joint_15)
            self.pose_tensor[16][:] = math.pi / 18 * torch.tanh(self.joint_16)
            
            self.pose_tensor[8][:] = math.pi / 18 * torch.tanh(self.joint_8)
            self.pose_tensor[9][:] = math.pi / 18 * torch.tanh(self.joint_9)
            self.pose_tensor[10][:] = math.pi / 18 * torch.tanh(self.joint_10)
            
            self.pose_tensor[11][:] = math.pi / 18 * torch.tanh(self.joint_11)
            self.pose_tensor[12][:] = math.pi / 18 * torch.tanh(self.joint_12)
            self.pose_tensor[13][:] = math.pi / 18 * torch.tanh(self.joint_13)
            
            self.pose_tensor[14][:] = math.pi / 18 * torch.tanh(self.joint_14)
            self.pose_tensor[15][:] = math.pi / 18 * torch.tanh(self.joint_15)
            self.pose_tensor[16][:] = math.pi / 18 * torch.tanh(self.joint_16)
            
            self.pose_tensor[21][:] = math.pi / 18 * torch.tanh(self.joint_21)
            self.pose_tensor[22][:] = math.pi / 18 * torch.tanh(self.joint_22)
            self.pose_tensor[23][:] = math.pi / 18 * torch.tanh(self.joint_23)
            
            self.pose_tensor[24][:] = math.pi / 18 * torch.tanh(self.joint_24)
            self.pose_tensor[25][:] = math.pi / 18 * torch.tanh(self.joint_25)
            self.pose_tensor[26][:] = math.pi / 18 * torch.tanh(self.joint_26)
            
            self.pose_tensor[27][:] = math.pi / 18 * torch.tanh(self.joint_27)
            self.pose_tensor[28][:] = math.pi / 18 * torch.tanh(self.joint_28)
            self.pose_tensor[29][:] = math.pi / 18 * torch.tanh(self.joint_29)
            
            self.pose_tensor[30][:] = math.pi / 18 * torch.tanh(self.joint_30)
            self.pose_tensor[31][:] = math.pi / 18 * torch.tanh(self.joint_31)
            
            self.pose_tensor[32][:] = math.pi / 18 * torch.tanh(self.joint_32)
            self.pose_tensor[33][:] = math.pi / 18 * torch.tanh(self.joint_33)
            '''
            # since the template is fully streched, some angle value can only be negative
        else: 
            
            self.pose_tensor[0][:] = math.pi / 1 * torch.tanh(self.joint_0)
            
            #self.pose_tensor[1][:] = math.pi / 9 * torch.tanh(self.joint_1)
            #self.pose_tensor[2][:] = math.pi / 9 * torch.tanh(self.joint_2)
            #self.pose_tensor[3][:] = math.pi / 18 * torch.tanh(self.joint_3)
            
            
            #self.pose_tensor[1][:] = math.pi / 4 * torch.tanh(self.joint_1)
            #self.pose_tensor[2][:] = math.pi / 9 * torch.tanh(self.joint_2)  # make the neck bone trainable(slightly)
            # for the shoulder using the symmetric deformation
            #self.pose_tensor[5][0] = math.pi / 3 * torch.tanh(self.joint_5[0][0])
            #self.pose_tensor[5][1] = math.pi / 3 * torch.tanh(self.joint_5[0][1])
            #self.pose_tensor[5][2] = math.pi / 3 * torch.tanh(self.joint_5[0][2])
            
            
            #self.pose_tensor[143][0] = math.pi / 3 * torch.tanh(self.joint_5[0][0])
            #self.pose_tensor[143][1] = -math.pi / 3 * torch.tanh(self.joint_5[0][1])
            #self.pose_tensor[143][2] = -math.pi / 3 * torch.tanh(self.joint_5[0][2])
            
           
            self.pose_tensor[4][2] = (  math.pi / 3 * torch.tanh(self.joint_4[0][2]))
            self.pose_tensor[142][2] = (  -math.pi / 3 * torch.tanh(self.joint_4[0][2]))
            

            #self.pose_tensor[6][:] = math.pi / 3 * torch.tanh(self.joint_6)
            #self.pose_tensor[144][:] = math.pi / 3 * torch.tanh(self.joint_144)
            #self.pose_tensor[6][0] = ( math.pi / 3 * torch.tanh(self.joint_6[0][0]))
            #self.pose_tensor[6][1] = (  math.pi / 3 * torch.tanh(self.joint_6[0][1]))
            #self.pose_tensor[6][2] = (  math.pi / 3 * torch.tanh(self.joint_6[0][2]))
            
            
            
            #self.pose_tensor[144][0] = (  math.pi / 3 * torch.tanh(self.joint_6[0][0]))
            #self.pose_tensor[144][1] = (  -math.pi / 3 * torch.tanh(self.joint_6[0][1]))
            #self.pose_tensor[144][2] = (  -math.pi / 3 * torch.tanh(self.joint_6[0][2]))
            
            #self.pose_tensor[7][1] = math.pi / 10 # * torch.tanh(self.joint_66[1])
            
            #self.pose_tensor[280][0] = (  -math.pi / 3 )#* torch.tanh(self.joint_6[0][0]))
            self.pose_tensor[69] = math.pi / 6 * torch.tanh(self.joint_69)
            self.pose_tensor[78] = math.pi / 6 * torch.tanh(self.joint_78)
            self.pose_tensor[87] = math.pi / 6 * torch.tanh(self.joint_87)
            self.pose_tensor[96] = math.pi / 6 * torch.tanh(self.joint_96)
            self.pose_tensor[105] = math.pi / 6 * torch.tanh(self.joint_105)
            self.pose_tensor[114] = math.pi / 6 * torch.tanh(self.joint_114)
            self.pose_tensor[123] = math.pi / 6 * torch.tanh(self.joint_123)
            self.pose_tensor[132] = math.pi / 6 * torch.tanh(self.joint_132)
            
        #self.pose_tensor[1] = self.joint_0
        
        #self.pose_tensor = self.pose_tensor.unsqueeze(0)
        # model will deform the mesh and then add the predetermined offset and learned displacement
        # apply the small adjustment on template first
        
        
        
        vertices = self.vertices # + 0.1 * torch.tanh(self.local_adjust.cuda())
       
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
        
        #flatten_loss = self.flatten_loss(vertices).mean()
        # add l2 regularization for small wing bones
        skining_matrix_adjust_l2 = torch.norm(self.skining_adjust)
        # define the return package, including the pose_tensor, vertices, faces, joints location, initial displacement, and local positon adjustment
        
        
        # define a new regulaization term to prevent the wing folder crazy
        # make sure the distance between 23 26 and 26 29 are the same
        # make sure the distance between 10, 13 and 13, 16 are the same
        distance_1 = joints[0][23] - joints[0][26]
        distance_2 = joints[0][29] - joints[0][26]
        reg_1 = torch.abs(torch.norm(distance_1) - torch.norm(distance_2))
        distance_1 = joints[0][10] - joints[0][13]
        distance_2 = joints[0][16] - joints[0][13]
        
        reg_2 = torch.abs(torch.norm(distance_1) -  torch.norm(distance_2))
        
        reg = reg_1 + reg_2 
            
        # assign weight for bone angle that for manuever that the bone with more muscles should move more
        # right now, classify the bones into three categories based on their muscle groups
        # weights: class 1: 0.1
        #          class 2: 0.3
        #          class 3: 0.5
        # measured with l2 norm
        # with loss function like this, the model will encourage bone that close to body deform the most
        
        # level 1 bone 4, 17, 5, 18
        # level 2 bone 6, 19 , 30, 32
        # level 3 bone 8, 11, 14, 21, 24, 27, 31, 33
        
        bone_prior_1 = 0.1 * (torch.norm(self.pose_tensor[4]) + torch.norm(self.pose_tensor[17])+torch.norm(self.pose_tensor[5])+torch.norm(self.pose_tensor[18]))
        
                                                                                                           
        bone_prior_2 = 0.2 * (torch.norm(self.pose_tensor[6]) + torch.norm(self.pose_tensor[19]) + torch.norm(self.pose_tensor[30]) + torch.norm(self.pose_tensor[32]))
                     
        bone_prior_3 = 0.3 * (torch.norm(self.pose_tensor[8]) 
                              + torch.norm(self.pose_tensor[11])
                              +torch.norm(self.pose_tensor[14])
                              +torch.norm(self.pose_tensor[21])
                              +torch.norm(self.pose_tensor[24])
                              +torch.norm(self.pose_tensor[27])
                              +torch.norm(self.pose_tensor[31])
                              +torch.norm(self.pose_tensor[33]))                                                                                                   
       
        
        # define a symmetric loss for each level of bones
        # the x axis rotation in the same direction while the y and z rotate in the opposite direction
        bone_prior = bone_prior_1 + bone_prior_2 + bone_prior_3
        bone_symmetric_1 = torch.norm(self.pose_tensor[5][0] - self.pose_tensor[18][0]) + \
                           torch.norm(self.pose_tensor[5][1] + self.pose_tensor[18][1]) + \
                           torch.norm(self.pose_tensor[5][2] + self.pose_tensor[18][2]) 
                           
        bone_symmetric_2 = torch.norm(self.pose_tensor[6][0] - self.pose_tensor[19][0]) + \
                           torch.norm(self.pose_tensor[6][1] + self.pose_tensor[19][1]) + \
                           torch.norm(self.pose_tensor[6][2] + self.pose_tensor[19][2])
          
                          
        bone_symmetric = 0.5 * bone_symmetric_1 + 0.2 * bone_symmetric_2
        
        #laplacian_loss = self.laplacian_smoothing(vertices).mean()
        return sr.Mesh(vertices.repeat(batch_size, 1, 1),self.faces.repeat(batch_size, 1, 1)), \
                       0, \
                       reg, \
                       bone_prior, \
                       bone_symmetric, \
                       self.pose_tensor, \
                       self.scale,       \
                       self.displacement, \
                       self.local_adjust,  \
                       vertices,            \
                       joints,              \
                       self.rest_distance, \
                       self.membrane_tip, \
                       estimated_location[0] + 0.1 * torch.tanh(self.displacement), \
                       self.training_skining_weight
                       
'''
IOU loss define the  
'''                  
def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union ).sum() / intersect.nelement()

def spring_force_loss(rest_distance, joints, index_list, spring_constant = 10): 
    
    spring_loss = 0
    
    new_distance = []
    
    for counter in range(len(index_list)-1): 
        first_joint_index = index_list[counter]
        second_joint_index = index_list[counter + 1]
        first_joint_tip = joints[0][first_joint_index]
        second_joint_tip = joints[0][second_joint_index]
        new_distance.append(torch.norm(second_joint_tip - first_joint_tip)) # restore back to original size
    
    spring_loss = 0
    force_list = []
    
    for counter in range(len(index_list)-1): 
        
        force_list.append(torch.abs(rest_distance[counter] - new_distance[counter]) * spring_constant)
    force_list_tensor = torch.FloatTensor(force_list)

    force_list_std = torch.std_mean(force_list_tensor)
    
    return force_list_std[0], force_list_std[1]
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
    pose_index = passed_pose_index
    image_size = (1024,1280)
    output_path = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/'.format(test_name, pose_index)
    #camera_list = [1]
    camera_meta_path = 'G:/PhDProject_real_data/{}/rearrange_pose/'.format(test_name)
    camera_list_path = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/'.format(test_name, pose_index)
    silouettee_image_path = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/'.format(test_name, pose_index)
    #estimated_location_file = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/estimated_location.txt'.format(test_name, pose_index)
    args.output_dir  = 'G:/PhDProject_real_data/{}/reconstruction/'.format(test_name)
    #args.image_output_dir = 'G:/PhDProject_real_data/{}/reconstruction/'.format(test_name)
    current_pose = pose_index
    # if use_previous == True
    # load the previous pose matrix as a starting point for the current pose reconstruction
    dataset = image_dataset(camera_meta_path, camera_list_path, silouettee_image_path, current_pose, use_previous)
    
    batch_size =dataset.camera_number 
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    
    #return

    model = Model(args.template_mesh).cuda()
    

    optimizer = torch.optim.Adam(model.parameters(), 0.005,betas=(0.5, 0.99))

    #renderer.transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

    epoch = tqdm.tqdm(list(range(0,epoch)))
    gif_images = []
    writer = imageio.get_writer(os.path.join('./', 'deform_bat_{}.gif'.format(pose_index)), mode='I',loop=0)
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
                        
            mesh, laplacian_loss,wing_tip_reg, bone_prior, bone_symmetric, current_pose, scale, displacement, local_adjust, vertices, joints,rest_distance, membrane_tip_index, displacement,weight_tensor = model(batch_size, estimated_location, use_previous, prev_pose[0])
            
            
            
            
            spring_loss_std, spring_loss_mean = spring_force_loss(rest_distance, joints, membrane_tip_index)
            renderer = sr.SoftRenderer(image_height=image_size[0], image_width=image_size[1],sigma_val=1e-6,
                                       camera_mode='projection', P = camera_matrix ,orig_height=image_size[0], orig_width=image_size[1], 
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
            #print("Laplacian_loss: ", 0.0001 * laplacian_loss)
            pose_loss = torch.tensor(0)
  
            if(use_previous == True):
                # only the body orientation is considered
                pose_loss = 0.005 * torch.norm(current_pose[:][:] - prev_pose[:][:]) #+ 0.1 * torch.norm(current_pose[1:][:] - prev_pose[1:][:])
            loss = IOU_loss + pose_loss  + 500 * laplacian_loss + 0 * wing_tip_reg  + 0.00 * bone_prior + 0.000 * bone_symmetric + 100 * spring_loss_std + 100*spring_loss_mean
            epoch.set_description('IOU Loss: %.4f   Pose Loss: %.4f  Wingtip_reg: %.4f  Bone prior: %.4f  Bone symmetry: %.4f  Spring_loss: %.4f' % (IOU_loss.item(),pose_loss.item(), wing_tip_reg.item(), 0.0 * bone_prior.item(), bone_symmetric.item(), spring_loss_mean.item()))
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        '''
        if i % 1 == 0:
            #print("pred_image shape: ",images_pred.detach().cpu().numpy()[0].shape )
            image = images_pred.detach().cpu().numpy()[0].transpose((1 , 2, 0))
            writer.append_data((255*image[:, :, 0]).astype(np.uint8))
            
            for counter in range(1):
                
                image = images_pred.detach().cpu().numpy()[3].transpose((1 , 2, 0))
                imageio.imsave(os.path.join(args.image_output_dir, 'pred_camera_{}_{}.png'.format(i, 2)), (255*image[..., 1]).astype(np.uint8))
            image_gt = images_gt.detach().cpu().numpy()[3].transpose((1, 2, 0))
        '''
    output_mesh,laplacian_loss, wing_tip_reg,bone_prior, bone_symmetric, current_pose, current_scale, current_displacement, local_adjust, vertices, joints,rest_distance,membrane_tip_list, displacement,weight_tensor = model(1, estimated_location,use_previous, prev_pose[0])
    output_mesh.save_obj(os.path.join(args.output_dir, '{}_bat_{}.obj'.format(test_name, pose_index)), save_texture=False)
    current_pose = current_pose.cpu().detach().numpy().tolist()
    current_scale = current_scale.cpu().detach().item()
    current_displacement = current_displacement.cpu().detach().numpy().tolist()
    local_adjust = local_adjust.cpu().detach().numpy().tolist()
    vertices = vertices.cpu().detach().numpy()
    vertices_mean = np.mean(vertices[0], axis=0)
    joints = joints.cpu().detach().numpy().tolist()
    template_displacement = displacement[0].cpu().detach().numpy().tolist()[0]
    output_skining_weight =weight_tensor[0].cpu().detach().numpy().tolist()
    #print(output_skining_weight.shape)
    #print("Displacement: ", template_displacement)
    save_dict = {"pose":current_pose,
                 "joints": joints, 
                 "scale":current_scale, 
                 "displacement":current_displacement, 
                 "local_adjust":local_adjust, 
                 "pose_loss":pose_loss.cpu().detach().item(), 
                 "IOU":IOU_loss.cpu().detach().item(),
                 "vertices_mean": vertices_mean.tolist(),
                 "template_displacement":template_displacement,
                 "skining_tensor": output_skining_weight}
    with open(os.path.join(silouettee_image_path, "output_membrane.json"), "w") as output:
        json.dump(save_dict, output)
        
    

if __name__ == '__main__':
    start_pose = 600
    end_pose = 600
    epoch = 100
    interval = 1
    test_name = "Brunei_2023_bat_test_13_1"
    for pose_index in range(start_pose, end_pose + interval, interval):
        print("working on pose: ", pose_index)

        # developing the use_previous to provide extra supervision
        main(test_name, pose_index, epoch, use_previous = False)
    
    
    