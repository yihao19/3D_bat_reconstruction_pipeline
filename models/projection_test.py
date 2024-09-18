# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:11:45 2023

@author: 18505
"""
import numpy as np

import cv2 as cv
import pickle
import torch
import math
import matplotlib.pyplot as plt
import soft_renderer as sr
# read the data template and project the vertices to the image 
# plane and  see if the camera matrix is right
bone_skining_matrix_path = './params.pkl'
with open(bone_skining_matrix_path, 'rb') as f:
     data = pickle.load(f)



template_mesh = sr.Mesh.from_obj('./4400.obj')
template_mesh_next = sr.Mesh.from_obj('./4400.obj')
vertices = template_mesh.vertices.cpu().numpy()[0]
vertices_next = template_mesh_next.vertices.cpu().numpy()[0]


vertices = data['v_template']
vertices_next = data['v_template']
print(vertices)
test_index = 2
'''
function that will extract the coordinate of the mesh vertices.
return the numpy arrray of the coordinates 
'''
def read_obj_vertices(obj_file):
    obj_file = open(obj_file)
    lines = obj_file.readlines()
    num_vertices = 0
    for line in lines: 
        parts = line.split(' ')
        if(parts[0] == 'v'):
            num_vertices += 1
        
    vertices = np.zeros((num_vertices, 3))
    counter = 0
    for line in lines: 
        parts = line.split(' ')
        if(parts[0] == 'v'):
            vertices[counter][0] = float(parts[1])
            vertices[counter][1] = float(parts[2])
            vertices[counter][2] = float(parts[3])
            counter += 1
    return vertices

def projection_test(vertices):
    ones = np.ones((vertices.shape[0], 1))
    homo_coordinates = np.concatenate((vertices, ones), axis = 1)
    
    homo_coordinates = np.expand_dims(homo_coordinates, axis=2)
    camera_matrix = np.loadtxt('./masks/camera_meta.txt')
    
   
    camera = camera_matrix[1].reshape((3, 4))
    
    camera = np.expand_dims(camera, axis=0)
    
    camera = camera.repeat(626, axis=0)
    
    coordinates = torch.bmm(torch.tensor(camera),torch.tensor(homo_coordinates))
    
    coordinates = coordinates.squeeze().cpu().numpy()
    
    
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    x_ = x / (z + 1e-5)
    y_ = y / (z + 1e-5)
    
    return x_, y_
def cartToPol(x, y):
      ang = np.arctan2(y, x)
      mag = np.hypot(x, y)
      return mag, ang
def dense_optical_flow(first_frame_path, second_frame_path):
    first_frame = cv.imread(first_frame_path)
    second_frame = cv.imread(second_frame_path)
    
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    next_gray = cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, 
                                       None,
                                       0.5, 3,5, 3, 5, 1.2, 0)
    

    # Computes the magnitude and angle of the 2D vectors
    # angle is where the pixel is moving 
    # magnitude is how many pixels it is moving
    mask = np.zeros_like(first_frame)
      
    # Sets image saturation to maximum
    mask[..., 1] = 255
    magnitude, angle = cartToPol(flow[..., 0], flow[..., 1])

    
    
    
    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    mask[..., 2] = magnitude
    
    fig, axes = plt.subplots(nrows=1, ncols=2)

    # generate randomly populated arrays

    
    # find minimum of minima & maximum of maxima
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    im1 = axes[0].imshow(angle)
    #fig.colorbar(im1, cax=cbar_ax)
    #fig.colorbar(im1, cax=cbar_ax)
    im2 = axes[1].imshow(magnitude)
    fig.colorbar(im2, cax=cbar_ax)
    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    
    return flow
    
first_index = 1
second_index = 5
obj_path = "C:\\Users\\18505\\SoftRas\\mesh_sequence\\pose_00000{}.obj".format(first_index)

next_path = "C:\\Users\\18505\\SoftRas\\mesh_sequence\\pose_00000{}.obj".format(second_index)


first_frame_path = "C:\\Users\\18505\\SoftRas\\models\\masks\\000{}.png".format(first_index)
second_frame_path = "C:\\Users\\18505\\SoftRas\\models\\masks\\000{}.png".format(second_index)


flow = dense_optical_flow(first_frame_path, second_frame_path)


vertices = read_obj_vertices(obj_path)
vertices_next = read_obj_vertices(next_path)


x_, y_ = projection_test(vertices)
x_flow, y_flow = projection_test(vertices_next) # get the next veritices position
# calculate the angles and magnitude of the optical flow of the vertices 

# calculate the magnitude and angle of the motion
# magnitude
x_mag = x_flow - x_
y_mag = y_flow - y_

total_mag = []
total_angle = []
for index in range(len(x_mag)):
    mag = math.sqrt(x_mag[index]**2 + y_mag[index]**2)
    angle = math.atan2(y_mag[index], x_mag[index]) + math.pi
    total_mag.append(mag)
    total_angle.append(angle)

frame_index = 1
image = cv.imread('./masks/000{}.png'.format(frame_index))
for counter in range(len(x_)):
        
    x = x_[counter]
    y = y_[counter]
    x = int(x)
    y = int(y)
    image[y][x] = (255, 0, 0)

    x__ = x_flow[counter]
    y__ = y_flow[counter]
    
    x_move = int(flow[y, x, 0])
    y_move = int(flow[y, x, 1])
    print(x_move, y_move)
    x__ = int(x__)
    y__ = int(y__)
    cv.line(image, (x, y), (x + x_move, y + y_move), (255, 0, 0), 1) 
cv.imwrite("./masks/camera{}_projected.png".format(frame_index),image)
image = cv.imread('./masks/000{}.png'.format(frame_index))
for counter in range(len(x_)):
        
    x = x_[counter]
    y = y_[counter]
    x = int(x)
    y = int(y)
    image[y][x] = (255, 0, 0)

    x__ = x_flow[counter]
    y__ = y_flow[counter]
    
    x_move = int(flow[y, x, 0])
    y_move = int(flow[y, x, 1])
    print(x_move, y_move)
    x__ = int(x__)
    y__ = int(y__)
    cv.line(image, (x, y), (x__, y__), (255, 0, 0), 1) 
cv.imwrite("./masks/camera{}_projected_gt.png".format(frame_index),image)










