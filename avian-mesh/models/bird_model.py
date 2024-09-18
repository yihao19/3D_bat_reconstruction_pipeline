import os
import json
import torch
import pickle
import cv2 as cv
from LBS import LBS
import matplotlib.pyplot as plt
from math import sin, cos
import numpy as np
from torch.nn import functional as F
import imageio
from pixels2svg import pixels2svg
'''
load the bat model:  
'''
class bird_model():
    '''
    Implementation of skined linear bird model
    '''
    def __init__(self, device=torch.device('cpu'), mesh='bird_fly_eccv.json'):
        
        self.device = device
        with open('./new_bat_params_version2_backforward.pkl', 'rb') as f:
            data = pickle.load(f)
        kintree_table = np.array([[ -1, 0, 1, 2, 2, 4, 5, 6, 7, 8, 9,  7,  11, 12, 7 , 14, 15, 2,  17, 18, 19, 20, 21, 22, 20, 24, 25, 20, 27, 28, 0,  30, 0,  32],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]])
        joints = data['joints_matrix'][3:, :].transpose()
        
        self.kintree_table = torch.tensor(kintree_table).to(device)
        self.parents = self.kintree_table[0].type(torch.LongTensor)
        self.weights = torch.tensor(data['weights']).to(device)
        

        
        #self.vert2kpt = torch.tensor(dd['vert2kpt']).to(device)
        # apply some random rotation

        
       
        self.J = torch.tensor(joints).unsqueeze(0).to(device) # the joint 
        self.V = torch.tensor(data['v_template']).unsqueeze(0).to(device) # the vertex
        random_dis = torch.tensor([0.00, 0.00, 0.00]) # randomly assign offset for 
        # vertices add to the place where the bat is roughly at
        self.V = self.V + random_dis.repeat(1, self.V.shape[1], 1).to(device)
        self.J = self.J + random_dis.repeat(1, self.J.shape[1], 1).to(device)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
       
        ax.axes.set_xlim3d(left=-5, right=5) 
        ax.axes.set_ylim3d(bottom=-3, top=3) 
        ax.axes.set_zlim3d(bottom=-5, top=5) 
        jx = data['v_template'][:, 0]
        jy = data['v_template'][:, 1]
        jz = data['v_template'][:, 2]       
        ax.scatter(jx,jy,jz,color='b') 
        
        self.LBS = LBS(self.J, self.parents, self.weights)
        
    def __call__(self, body_pose, pose2rot=True):
        batch_size = body_pose.shape[0]
        V = self.V.repeat([batch_size, 1, 1])
        J = self.J.repeat([batch_size, 1, 1])
        # concatenate bone and pose
        #bone = torch.cat([torch.ones([batch_size,1]).to(self.device), bone_length], dim=1)
        pose = body_pose
       
        # LBS       
        # call it 
        verts, joints = self.LBS(V,J,  pose, to_rotmats=pose2rot)
        # this projection doesn't need key point 
        return verts, joints



def track_to_root(kintree_table, joint_rotation_matrices, child_index, result):
    if(kintree_table[0][child_index] == -1):
        return 
    # times the parent 
    parent_index = kintree_table[0][child_index]
    
    child_index = parent_index
    result = result @ joint_rotation_matrices[parent_index].transpose()
    track_to_root(kintree_table, joint_rotation_matrices, child_index, result)

    
if __name__=="__main__":
    
    number_bone = 34
    model = bird_model()
    counter = 0
    scale = 0.2
    move = 20
    # define the pose matrix
    writer = imageio.get_writer('deform.gif', mode='?', loop=0)
    radiant = 0
    while(radiant < 1):
        
        joint_indexes = [0, 5, 18]
        pose = np.zeros((number_bone, 3))
        
        for joint_index in joint_indexes:
            angle = radiant
            if(joint_index == joint_indexes[0]):
                pose[joint_index][0] = 0
                pose[joint_index][1] = 0
                pose[joint_index][2] = 0
            elif(joint_index == joint_indexes[1]):
            
                pose[joint_index][0] = 0
                pose[joint_index][1] = 0
                pose[joint_index][2] = -angle
            elif(joint_index == joint_indexes[2]):
                pose[joint_index][0] = 0
                pose[joint_index][1] = 0
                pose[joint_index][2] = angle
        
        pose = torch.tensor(pose)
        pose = pose.unsqueeze(0)
                
        
        vertices, joints = model(pose, pose2rot=True)
        
        joints = joints.squeeze().cpu().numpy()
        vertices = vertices.squeeze().cpu().numpy()
        
        

        np.savetxt("vertices.txt", vertices)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        
        ax.axes.set_xlim3d(left=19, right=21) 
        ax.axes.set_ylim3d(bottom=19, top=21) 
        ax.axes.set_zlim3d(bottom=19, top=21) 
        
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.zaxis.set_tick_params(labelbottom=False)

        jx = vertices[:, 0]
        jy = vertices[:, 1]
        jz = vertices[:, 2]

        radiant += 0.1
        #ax.scatter(scale * jx+move,scale * jy+move,scale * jz+move,color='black') 

        table = model.kintree_table.squeeze().cpu().numpy()
        
        for index in range(number_bone):
            if(table[0][index] == -1):
                continue
            else:
                parent_index = table[0][index]
                child_index = table[1][index]
                parent = joints[parent_index]
                child = joints[child_index]
                x, y, z = [scale * parent[0]+move, scale * child[0] + move], [scale * parent[1]+move, scale * child[1]+ move], [scale * parent[2]+ move, scale*child[2]+move]
                #ax.text(joints[child_index][0],joints[child_index][1],joints[child_index][2], str(index)) 
                ax.view_init(elev=30., azim=-60)
                ax.plot(x, y, z, color='black')
        
        fig.savefig('skeleton_{}_rotation.svg'.format(counter))
        counter += 1
        #plt.close()
        #image = cv.imread('my_plot_{}.png'.format(counter))
        #pixels2svg('my_plot_{}.png'.format(counter),'my_plot_{}.svg'.format(counter))
        #counter += 1
        #writer.append_data(image)
        
    writer.close()

    '''
    
    # check all the parameters
    # vertices and 3D plot it
    joints = model.J.squeeze().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(x, y, z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    jx = joints[:, 0]
    jy = joints[:, 1]
    jz = joints[:, 2]

    
    # draw the line of skeleton to verify the understanding
    
    
    # trying to apply the rotation matrix to update the 
    # joint coordinates
    # generate the rotation matrix for each pose
    pitches = np.zeros(25)
    yawes = np.zeros(25)
    rolls = np.zeros(25)
    joint_rotation_matrices =np.zeros((25, 3, 3))
    # trying to make the joint 20 rotate wrt joint 19
    child_index = 20
    pitches[child_index] = 0.5
    yawes[child_index] = 0.5
    rolls[child_index] = 0.5
  
    # construct the entire rotation matrix
    
    for joint_index in range(25):
        rotation_matrix = euler_angle(pitches[joint_index], yawes[joint_index], rolls[joint_index])
        joint_rotation_matrices[joint_index] = rotation_matrix

    
    # construct the rotation matrix from root to any node
    #
    kintree_table = model.kintree_table.cpu().numpy()
    print("kintree_table: ",kintree_table)
    global_rotation_matrices = np.zeros((4, 4))
    global_rotation_matrices[3, 3] = 1
    parent_index = kintree_table[0][child_index]
    print(parent_index)
    global_rotation_matrices[3, 0] = (jx[child_index] - jx[parent_index])
    global_rotation_matrices[3, 1] = (jy[child_index] - jy[parent_index])
    global_rotation_matrices[3, 2] = (jz[child_index] - jz[parent_index])
    initial_matrix = joint_rotation_matrices[child_index]
    
    
    track_to_root(kintree_table, joint_rotation_matrices, child_index, initial_matrix)
    

    global_rotation_matrices[:3, :3] = initial_matrix
    # plot the transformed skeleton
    vertex = np.array([[jx[child_index]], [jy[child_index]], [jz[child_index]], [1]])
    print(vertex.shape)
    result = global_rotation_matrices @ vertex
    print(result)
    jx = joints[:, 0]
    jy = joints[:, 1]
    jz = joints[:, 2]
    jx[child_index] = result[0] #/ result[3] * 1.0
    jy[child_index] = result[1] #/ result[3] * 1.0
    
    jz[child_index] = result[2] #/ result[3] * 1.0
    
    for index in range(25):
        ax.scatter(jx[index],jy[index],jz[index],color='b') 
        ax.text(jx[index],jy[index],jz[index], str(index)) 
    table = model.kintree_table.squeeze().cpu().numpy()
    for index in range(25):
        if(table[0][index] == -1):
            continue
        else:
            parent_index = table[0][index]
            child_index = table[1][index]
            parent = joints[parent_index]
            child = joints[child_index]
            x, y, z = [parent[0], child[0]], [parent[1], child[1]], [parent[2], child[2]]
            ax.plot(x, y, z, color='black')
    
    #plt.show()
    '''