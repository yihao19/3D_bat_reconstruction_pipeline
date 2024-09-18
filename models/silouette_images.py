# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:49:35 2023

@author: 18505
"""

import cv2 as cv
import numpy as np
import os
import torch
'''
for index in range(1, 2):
    
    input_folder = 'C:\\Users\\18505\\Desktop\\GT.png'
    
    image_number = 50
    
    camera_list = []
    for counter in range(1, image_number+1):
        if(counter < 10):
            image_name = "000"+str(counter)
        else:
            image_name = "00"+str(counter)
        image_path = os.path.join(input_folder, "camera{}".format(counter) + ".png")
        image = cv.imread(image_path)
        pixel_counter = 0
        
        for row in range(720):
            for col in range(720):
                if(image[row][col][0] > 200):
                    image[row][col] = [0,0,0]
                else:
                    image[row][col] = [255,255,255]
                    pixel_counter += 1
        if(pixel_counter > 10):
            camera_list.append(counter)
         
        
        output_image_name = os.path.join(input_folder+'/masks',"camera{}".format(counter)  + ".png")
        cv.imwrite(output_image_name,image)
        print("working on frame: {}, and camera:{}".format(index, counter))
    output_text_file = os.path.join(input_folder + '/masks', 'camera.txt')    
    file = open(output_text_file, 'w')
    file.write(str(camera_list))
    file.close()
'''
image_path = 'C:\\Users\\18505\\Desktop\\GT_1.png'
image = cv.imread(image_path)
for row in range(720):
    for col in range(720):
        if(image[row][col][0] > 200):
            image[row][col] = [0,0,0]
       
cv.imwrite(image_path,image)