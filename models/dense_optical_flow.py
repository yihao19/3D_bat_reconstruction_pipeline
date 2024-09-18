# -*- coding: utf-8 -*-
"""
purpose: this code will get the dense optical flow of a video from single 
         camera and return the optical flow mask
         
         Dense optical flow
"""

# make a video a
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt





# calculate the optical flow between two camera frame
first_index = 610
second_index = 615


camera = 41
first_image = "G:\\PhDProject_real_data\\brunei_2023_bat_test_13_1\\camera41\\camera41{}.png".format(first_index)
second_image = "G:\\PhDProject_real_data\\brunei_2023_bat_test_13_1\\camera41\\camera41{}.png".format(second_index)


first_image = cv.imread(first_image, cv.IMREAD_GRAYSCALE)
second_image = cv.imread(second_image, cv.IMREAD_GRAYSCALE)
flow = cv.calcOpticalFlowFarneback(first_image, second_image, 
                                       None,
                                       0.5, 3,20, 3, 5, 1.2, 0)
    
fig = plt.figure(figsize=(1, 2))

fig.add_subplot(1, 2, 1)
plt.imshow(flow[...,1])
plt.title("Y optical")
plt.colorbar(orientation='vertical')
fig.add_subplot(1, 2, 2)
plt.imshow(flow[...,0])
plt.title("X optical")
#plt.imshow(second_image[:, :,0])
plt.colorbar(orientation='vertical')