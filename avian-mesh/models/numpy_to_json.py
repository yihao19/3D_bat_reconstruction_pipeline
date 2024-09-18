# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:23:17 2023

@author: 18505
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# this function will convert the numpy array of vertices, joints, and weights to json format
with open('params.pkl', 'rb') as f:
    data = pickle.load(f)
    
    
for key in data.keys():
    print(key)



print(data['v_template'].shape)
x = data['v_template'][:, 0]
y = data['v_template'][:, 1]
z = data['v_template'][:, 2]

ax.axes.set_xlim3d(left=-0.5, right=0.5) 
ax.axes.set_ylim3d(bottom=-0.5, top=0.5) 
ax.axes.set_zlim3d(bottom=-0.5, top=0.5) 


#ax.scatter(x,y,z,color='r') 

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

joints = data['joints_matrix'][:3, :].transpose()

print(joints.shape)


jx = joints[:, 0]
jy = joints[:, 1]
jz = joints[:, 2]
#ax.scatter(jx,jy,jz,color='b') 

table = np.array([[-1, 0, 0, 1, 1, 4, 3, 5, 6, 7, 8,  5,  6,  11, 12, 5,  6 , 15, 16],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 ]])


for index in range(19):
    if(table[0][index] == -1):
        continue
    else:
        parent_index = table[0][index]
        child_index = table[1][index]
        parent = joints[parent_index]
        child = joints[child_index]
        x, y, z = [parent[0], child[0]], [parent[1], child[1]], [parent[2], child[2]]
        ax.plot(x, y, z, color='black')

for index in range(19):
    ax.scatter(jx[index],jy[index],jz[index],color='b') 
    ax.text(jx[index],jy[index],jz[index], str(index)) 



# skining matrix: 
weights = data["weights"]
print(weights.shape)
plt.show()