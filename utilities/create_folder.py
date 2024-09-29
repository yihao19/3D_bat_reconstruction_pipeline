# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:45:25 2021

@author: 18505
"""

import os
import shutil
# Directory
des_dir = "D:/PhDProject_real_data/brunei_2023_bat_test_17_1/rearrange_pose/"
src_dir = "D:/PhDProject_real_data/brunei_2023_bat_test_17_1/"

#text_file = "G:\\GCN_project\\Bat_Sample_Model\\new_skeleton_bat_data\\rearranged_data\\validation_list.txt"
# Parent Directory path


#file = open(text_file, 'w')
'''
for counter in range(200, 450, 1):
    folder_name = "{}".format(counter)
    #file.write(folder_name + ".xyz\n")
    folder = os.path.join(des_dir, folder_name)
    os.mkdir(folder)
#file.close()
'''
camera_index = {21:0, 23:1, 24:2, 25:3, 31:4, 32:5, 34:7, 35:8, 41:9, 42:10, 43:11, 44:12}
for camera in camera_index:
    print("working on camera: " + str(camera))
    for counter in range(200, 450, 1):

        if(counter < 10):
            fileName = " 000"+str(counter)+".png"
        elif(counter < 100):
            fileName = " 00"+str(counter)+".png"
        elif(counter < 1000):
            fileName = " 0"+str(counter)+".png"
        else:
            fileName = " "+str(counter)+".png"

        file_name = "camera{}{}.png".format(camera, counter)
        des_file_name = "camera{}.png".format(camera_index[camera] + 1)
        des_file = os.path.join(des_dir, "{}".format(counter), des_file_name)
        src_file = os.path.join(src_dir, "camera{}".format(camera), "masks", file_name)
        print(src_file)
        print(des_file)
        shutil.copy(src_file, des_file)
