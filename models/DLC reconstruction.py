# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:21:10 2023

@author: 18505
"""
'''
for each frame: 
    for each key point go through each camera: 
        if the point is seen in more than 2 cameras:
            reconstruct
'''
#deeplabcut file
import pandas
frame_index_start = 1
frame_index_end = 500
camera_number_threshold = 2
camera_index = 21
camera_list = [21,23,24,25,31,32,33,34,35,41,42,43,44]
test_file = "G:\\2023_DLC\\2023-camera{}-2023-08-28\\labeled-data\\camera{}\\CollectedData_camera{}.csv".format(camera_index, camera_index, camera_index)
key_point_list = ["", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
key_point_dict = dict()
for frame_index in range(frame_index_start, frame_index_end):
    for key_point in key_point_list: 
        for camera in camera_list: 
            # get the key_point
            if(key_point == ""):
                key_point_name = "camera{}".format(camera)
            else: 
                key_point_name = "camera{}.{}".format(camera, key_point)

    


