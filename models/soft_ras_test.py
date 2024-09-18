# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:41:19 2023

@author: 18505
"""

'''
soft rast render test

'''

"""
Demo render.
1. save / load textured .obj file
2. render using SoftRas with different sigma / gamma
"""
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

import soft_renderer.cuda.load_textures as load_textures_cuda

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')
'''
the texture model is used to train a texture map of the model
'''

class Model(nn.Module):
	def __init__(self, template_path):
		super(Model, self).__init__()

		# set template mesh
		self.template_mesh = sr.Mesh.from_obj(template_path, load_texture=True, texture_res=1)
		self.register_parameter('textures', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
	
	def forward(self):
		textures = (self.textures)
		return sr.Mesh(self.template_mesh.vertices.clone().repeat(1, 1, 1), 
                       self.template_mesh.faces.clone().repeat(1, 1, 1),
                       textures=textures.repeat(1, 1, 1), 
                       texture_res=1, texture_type='vertex')
        
'''
IOU loss define the  
'''                  
def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str,
                        default=os.path.join(data_dir, 'obj/spot/pose.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
                        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()
    camera_matrix_file = "C:/Users/18505/SoftRas/data/obj/camera_meta.txt"
    
    entire_camera_matrix = np.loadtxt(camera_matrix_file)
    camera_number = 31
    camera_matrix = np.reshape(entire_camera_matrix, (31,1, 3, 4)).astype("float32")
    
    # other settings
   

    # load from Wavefront .obj file
    
    # create renderer with SoftRas
    camera_index = 1

    model = Model(args.filename_input).cuda()
    

    image_size = (720, 720)
    optimizer = torch.optim.Adam(model.parameters(), 0.005,betas=(0.5, 0.99))
    loop = tqdm.tqdm(list(range(0, 500)))
    
    for epoch in loop:
        mesh = model()
        
        renderer = sr.SoftRenderer(image_height=image_size[0], image_width=image_size[1],sigma_val=1e-6,
                                   camera_mode='projection', P = camera_matrix[camera_index],orig_height=image_size[0], orig_width=image_size[1], 
                                   near=0, far=100)
        images = renderer(mesh.vertices, mesh.faces, mesh.textures, texture_type='vertex')[0,0:3,:,:].permute(1, 2, 0)
        
        #images = renderer.render_mesh(mesh_)
        gt_image_path = "C:\\Users\\18505\\SoftRas\\data\\obj\\spot\\GT_1.png"
        
        gt_image = cv.imread(gt_image_path,cv.COLOR_BGR2RGB) / 255.
        gt_image_tensor = torch.tensor(gt_image)

        
        gt_image_tensor = gt_image_tensor.unsqueeze(0).cuda() 

        
        loss = torch.norm(gt_image_tensor - images)
        loop.set_description('L2 Loss: %.4f  ' % (loss.item()))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if(epoch % 10 == 0):   
            image = images.detach().cpu().numpy()
            #mesh.save_obj(os.path.join(args.output_dir, 'saved_spot{}.obj'.format(epoch)), save_texture=False)
            cv.imwrite(os.path.join(args.output_dir, 'pred_camera_{}.png'.format(epoch)), (255 * image[..., 0:3]).astype(np.uint8))
    
    
    mesh.save_obj(os.path.join(args.output_dir, 'saved_spot.obj'), save_texture=True) 
    '''
    
    mesh = model()
    mesh.textures = mesh.textures
    mesh.save_obj(os.path.join(args.output_dir, 'saved_spot.obj'), save_texture=True)
    # draw object from different sigma and gamma
    '''
    output_mesh1 =  sr.Mesh.from_obj(os.path.join(data_dir, 'obj/spot/saved_spot.obj'), load_texture=True, texture_res=20)
    output_mesh2 =  sr.Mesh.from_obj(os.path.join(data_dir, 'obj/spot/pose_2.obj'), load_texture=True, texture_res=20)
    
    image_size = (720, 720)
    transform = sr.Transform("projection",P = camera_matrix[camera_index], orig_height = 720, orig_width = 720)
    lighting = sr.Lighting()
    renderer = sr.SoftRenderer(image_height=image_size[0], image_width=image_size[1],sigma_val=1e-6,
                               camera_mode='projection', P = camera_matrix[camera_index],orig_height=image_size[0], orig_width=image_size[1], 
                               near=0, far=100)
    #output_mesh = lighting(output_mesh1)
    #output_mesh = transform(output_mesh)
    images = renderer(output_mesh1.vertices, output_mesh1.faces, output_mesh1.textures)[0,0:3,:,:].permute(1, 2, 0)
    image = images.detach().cpu().numpy()
    cv.imwrite(os.path.join(args.output_dir, 'validate_{}_{}.png'.format(0, camera_index)), (255 * image[..., 0:3]).astype(np.uint8))
    # load the texture and model and validate the result
    
    mesh = sr.Mesh.from_obj(args.filename_input, load_texture=True, texture_type="vertex")
    
    mesh.textures = torch.zeros_like(mesh.vertices)
    renderer = sr.SoftRenderer(image_height=720, image_width=720,sigma_val=1e-6,
                               camera_mode='projection', P = camera_matrix[camera_index],orig_height=720, orig_width=720, 
                               near=0, far=100)
    renderer = sr.SoftRenderer(camera_mode="look_at", texture_type="vertex")
    renderer.transform.set_eyes_from_angles(-380, 0, 0)
    images = renderer.render_mesh(mesh)
    image = images.detach().cpu().numpy()[0].transpose((1,2,0))
    image = (255*image).astype(np.uint8)
    

    # save to textured obj
    
   

if __name__ == '__main__':
    main()