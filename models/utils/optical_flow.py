# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:20:25 2023
purpose: 
    generate forward and backward optical flow by giving the start and end index
    of images

@author: 18505
"""
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
'''
calculate the forward optical flow and return the sequence of optical flow in 
a numpy array
return shape: (# sequence, image_height, image_width, 2)
              last two dim(x_move, y_move)
'''

def forward_optical_flow(image_path, image_size, start_index, end_index):
    
    fw_flow = np.zeros((end_index - start_index, image_size[0], image_size[1], 2))
    index = start_index
    counter = 0
    while(index < end_index):
        
        prev_image = cv.imread(os.path.join(image_path, "000{}.png".format(index)))
        next_image = cv.imread(os.path.join(image_path, "000{}.png".format(index + 1)))
        prev_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
        next_gray = cv.cvtColor(next_image, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, 
                                           None,
                                           0.5, 3,5, 3, 5, 1.2, 0)
        fw_flow[counter] = flow
        index += 1
        counter += 1
        
    return fw_flow

def backward_optical_flow(image_path, image_size, start_index, end_index):
    
    bw_flow = np.zeros((end_index - start_index, image_size[0], image_size[1], 2))
    index = end_index
    counter = -1
    while(index > start_index):
        prev_image = cv.imread(os.path.join(image_path, "000{}.png".format(index)))
        next_image = cv.imread(os.path.join(image_path, "000{}.png".format(index - 1)))
        prev_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
        next_gray = cv.cvtColor(next_image, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, 
                                           None,
                                           0.5, 3,5, 3, 5, 1.2, 0)
        bw_flow[counter] = flow
        index -= 1
        counter -= 1
    return bw_flow


if __name__=="__main__":
    image_path = "C:\\Users\\18505\\SoftRas\\models\\masks\\"
    start_index = 1
    end_index = 5
    def cartToPol(x, y):
          ang = np.arctan2(y, x)
          mag = np.hypot(x, y)
          return mag, ang
    fw_flow = forward_optical_flow(image_path, (720, 720), start_index, end_index)
    bw_flow = backward_optical_flow(image_path, (720, 720), start_index, end_index)
    
    fig, axes = plt.subplots(nrows=1, ncols=2)

    # generate randomly populated arrays

    
    # find minimum of minima & maximum of maxima
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    im1 = axes[0].imshow(bw_flow[0,:,:,0])
    #fig.colorbar(im1, cax=cbar_ax)
    #fig.colorbar(im1, cax=cbar_ax)
    im2 = axes[1].imshow(bw_flow[0,:,:,1])
    fig.colorbar(im2, cax=cbar_ax)
    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    