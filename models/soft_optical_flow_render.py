# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 21:49:44 2023

@author: 18505
"""

'''
this function will reconstruct the optical flow between two consecutive pose
'''


#
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




# load the camera matrix
def main():
   
   
    camera_matrix_file = "G:\\PhDProject_real_data\\brunei_2023_bat_test_13_1\\rearrange_pose\\camera_meta.txt"
    reconstruction_folder = "G:\\PhDProject_real_data\\brunei_2023_bat_test_13_1\\reconstruction\\"
    test_name = "brunei_2023_bat_test_13_1_bat_"
    entire_camera_matrix = np.loadtxt(camera_matrix_file)
    camera_number = 13
    camera_matrix = np.reshape(entire_camera_matrix, (camera_number,1, 3, 4)).astype("float32")
    
    # other settings
    first_index = 610
    second_index = 615
    first_image = "G:\\PhDProject_real_data\\brunei_2023_bat_test_13_1\\camera41\\camera41{}.png".format(first_index)
    second_image = "G:\\PhDProject_real_data\\brunei_2023_bat_test_13_1\\camera41\\camera41{}.png".format(second_index)
    # load from Wavefront .obj file
    
    # create renderer with SoftRas
    camera_index = 9
    

    _first_image = cv.imread(first_image,cv.IMREAD_GRAYSCALE)
    _second_image = cv.imread(second_image,cv.IMREAD_GRAYSCALE)
    first_pose_file = os.path.join(reconstruction_folder, "{}{}.obj".format(test_name, first_index))
    second_pose_file = os.path.join(reconstruction_folder, "{}{}.obj".format(test_name, second_index))
    
    image_size = (1024, 1280)
    
    # load two mesh
    
    first_mesh = sr.Mesh.from_obj(first_pose_file)
    second_mesh = sr.Mesh.from_obj(second_pose_file)
    
    displacement = second_mesh.vertices - first_mesh.vertices
    
    
    print(displacement.shape)
    print(first_mesh.vertices.shape)
    renderer = sr.SoftRenderer(image_height=image_size[0], image_width=image_size[1],sigma_val=1e-6,
                               camera_mode='projection', P =np.concatenate((camera_matrix[camera_index],camera_matrix[camera_index])),orig_height=image_size[0], orig_width=image_size[1], 
                               near=0, far=100)
    #images = renderer(mesh.vertices, mesh.faces, mesh.textures, texture_type='vertex')[0,0:3,:,:].permute(1, 2, 0)
    # assign the displacement to the first mesh as vertex color
    
    flow_mesh =  sr.Mesh(torch.cat([first_mesh.vertices, first_mesh.vertices]), torch.cat([first_mesh.faces,first_mesh.faces]), torch.cat([first_mesh.vertices,second_mesh.vertices]), texture_type='vertex')
    #second_mesh = sr.Mesh(second_mesh.vertices, second_mesh.faces, second_mesh.vertices, texture_type='vertex')

    images = renderer(flow_mesh.vertices, flow_mesh.faces, flow_mesh.textures, texture_type='vertex')#[0,0:3,:,:].permute(1, 2, 0)
    #second_image = renderer(second_mesh.vertices, second_mesh.faces, second_mesh.textures, texture_type='vertex')[0,0:3,:,:].permute(1, 2, 0)
    print(images.shape)

    first_image = images[0, 0:3, :, :].permute(1, 2, 0)
    second_image = images[1, 0:3, :, :].permute(1, 2, 0)

    first_image = torch.cat([first_image, torch.ones(1024, 1280, 1).cuda()], dim=2)
    second_image = torch.cat([second_image, torch.ones(1024, 1280, 1).cuda()], dim=2)

    cam = torch.tensor(camera_matrix[camera_index].squeeze(0).transpose()).cuda()
    pro_first_image = torch.matmul(first_image, cam)
    pro_second_image = torch.matmul(second_image,cam)
    
    pro_first_image[...,0] = pro_first_image[..., 0] / pro_first_image[..., 2]
    pro_first_image[...,1] = pro_first_image[..., 1] / pro_first_image[..., 2]
    
    pro_second_image[...,0] = pro_second_image[..., 0] / pro_second_image[..., 2]
    pro_second_image[...,1] = pro_second_image[..., 1] / pro_second_image[..., 2]

    fw_flow = pro_second_image - pro_first_image
    
   
    flow = cv.calcOpticalFlowFarneback(_first_image, _second_image, 
                                           None,
                                           0.5, 3,30, 3, 5, 1.2, 0)
    #cv.imwrite('validate_{}_{}.png'.format(0, camera_index), (255 * image[..., 0:3]).astype(np.uint8))
    fig = plt.figure(figsize=(3, 2))
    
    fw_flow = fw_flow.cpu().numpy()
    fig.add_subplot(3, 2, 1)

    plt.imshow(fw_flow[...,1])
    plt.title("Y optical")
    plt.colorbar(orientation='vertical')
    fig.add_subplot(3, 2, 2)
    plt.imshow(fw_flow[...,0])
    plt.title("X optical")
    #plt.imshow(second_image[:, :,0])
    plt.colorbar(orientation='vertical')
    fig.add_subplot(3, 2, 3)
    plt.imshow(_first_image)
    plt.colorbar(orientation='vertical')
    plt.title("original first {}".format(first_index))
    fig.add_subplot(3, 2, 4)
    plt.imshow(_second_image)
    plt.colorbar(orientation='vertical')
    plt.title("original second {}".format(second_index))
    fig.add_subplot(3, 2, 5)
    plt.imshow(flow[...,1])
    plt.colorbar(orientation='vertical')
    plt.title("Y optical from original")
    fig.add_subplot(3, 2, 6)
    plt.imshow(flow[...,0])
    plt.colorbar(orientation='vertical')
    plt.title("x optical from original")
    #plt.show()
    return 
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
