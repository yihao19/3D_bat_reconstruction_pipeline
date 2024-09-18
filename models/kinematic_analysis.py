# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:21:17 2023

@author: 18505
"""

import numpy as np

import json
# read the output json file 

import os
from matplotlib import pyplot as plt



def read_pose_json(json_file, bone_index):
    output_json = 1
    f = open(json_file)
    output_json = json.load(f)
    
    pose_array = output_json['pose']
    print(output_json['scale'])
    output_json_x = []
    output_json_y = []
    output_json_z = []
    for bone in bone_index: 
        output_json_x.append(pose_array[bone][0])
        output_json_y.append(pose_array[bone][1])
        output_json_z.append(pose_array[bone][2])
    return output_json_x, output_json_y, output_json_z

if __name__=="__main__":
    print("Hello")
    path = "D:\\PhDProject_real_data\\" 
    test_case = "brunei_2023_bat_16"
    
    output_path = os.path.join(path, test_case, "rearrange_pose")
    print(output_path)
    start_index =  551
    end_index = 1400
    bone_index = [0, 7, 20]
    output_jsons_x = []
    output_jsons_y = []
    output_jsons_z = []
    for counter in range(start_index, end_index+1): 
        json_path = os.path.join(output_path, str(counter), "output.json")
        output_json_x,output_json_y, output_json_z = read_pose_json(json_path, bone_index)
       
        output_jsons_x.append(output_json_x)
        output_jsons_y.append(output_json_y)
        output_jsons_z.append(output_json_z)
    
    
    output_json_x_array = np.array(output_jsons_x)
    output_json_y_array = np.array(output_jsons_y)
    output_json_z_array = np.array(output_jsons_z)
    fig = plt.figure()
    
    plt.title("{}-Yaw(X axis)".format(test_case))
    plt.xlabel("Frame index")
    plt.ylabel("rotation in radiant")
    plt.plot(output_json_x_array[:, 1])
    plt.plot(output_json_x_array[:, 2])
    plt.legend(["Bone: {}".format(bone_index[1]), "Bone: {}".format(bone_index[2])])
    
    plt.savefig("X")
    fig = plt.figure()
    plt.title("{}-Yaw(Y axis)".format(test_case))
    #plt.plot(output_json_array[:, 0])
    plt.xlabel("Frame index")
    plt.ylabel("rotation in radiant")
   
    plt.plot(output_json_y_array[:, 1])
    plt.plot(output_json_y_array[:, 2])
    plt.legend(["Bone: {}".format(bone_index[1]), "Bone: {}".format(bone_index[2])])
    plt.savefig("Y")
    fig = plt.figure()
    plt.title("{}-Yaw(Z axis)".format(test_case))
    #plt.plot(output_json_array[:, 0])
    plt.xlabel("Frame index")
    plt.ylabel("rotation in radiant")
 
    plt.plot(output_json_z_array[:, 1])
    plt.plot(output_json_z_array[:, 2])
    plt.legend(["Bone: {}".format(bone_index[1]), "Bone: {}".format(bone_index[2])])
    plt.savefig("Z")
    
    
    plt.show()
        
    
    
    
    