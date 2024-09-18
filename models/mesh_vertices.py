# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:27:05 2023
puspose: read the mesh and calculate the average location of the current pose
@author: 18505
"""

import numpy as np
import os

def point_cloud(): 
    total_pose = 1351
    path = "G:\\GCN_project\\Bat_Sample_Model\\high_speed_tunnel\\rearranged\\mesh_sequence\\"
    des = "G:\\GCN_project\\Bat_Sample_Model\\high_speed_tunnel\\rearranged\\"
    for pose in range(1, total_pose + 1):
        name = ""
        if(pose < 10):
            name = "mesh_00000{}.obj".format(pose)
        elif(pose < 100):
            name = "mesh_0000{}.obj".format(pose)
        else: 
            name = "mesh_000{}.obj".format(pose)
        obj_file = open(os.path.join(path, name))
        counter = 0
        x_total = 0
        y_total = 0
        z_total = 0
        lines = obj_file.readlines()
        point_cloud = []
        # also store the point cloud as another 3D supervision
        
        for line in lines: 
            parts = line.split(' ')
            if(parts[0] != 'v'):
                continue
            else: 
                counter += 1
                x_total += float(parts[1])
                y_total += float(parts[2])
                z_total += float(parts[3])
                point_cloud.append([float(parts[1]), float(parts[2]), float(parts[3])])
        x_average = x_total / counter
        y_average = y_total / counter
        z_average = z_total / counter
        estimated_point_cloud_array = np.array(point_cloud)
    
        des_file = os.path.join(des, "pose{}".format(pose), "masks/estimated_location.txt")
        point_cloud_file = os.path.join(des, "pose{}".format(pose), "masks/estimated_point_cloud.txt")
        txt_file = open(des_file, 'w')
        
        txt_file.write("{} {} {}".format(x_average, y_average, z_average))
        
        np.savetxt(point_cloud_file, estimated_point_cloud_array)
    
        txt_file.close()

'''
visualize the point cloud 
'''
def visualize(point_cloud_txt):
    
    
    
    return 

if __name__ == "__main__":
    print("Hello")

    # adding some noise to the point cloud 
    

























