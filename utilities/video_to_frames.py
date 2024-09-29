# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:54:50 2022

@author: 18505
"""


# Program To Read video
# and Extract Frames
import cv2
import os 

# Function to extract frames
def FrameCapture(video_path, image_saving_path, camera_name):
    if(os.path.exists(image_saving_path) == False):
        os.mkdir(image_saving_path)
        
    for image in os.listdir(image_saving_path):
        if(image.endswith('.png')):
            os.remove(os.path.join(image_saving_path, image))  
    # Path to video file
    vidObj = cv2.VideoCapture(video_path)
  
    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success = 1
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if(count >= 5000 and count <= 5800   and count % 1== 0):
            
        # Saves the frames with frame-count
            cv2.imwrite(image_saving_path + "camera{}{}.png".format(camera_name, count), image)
      
        count += 1  
        
  
# Driver Code
if __name__ == '__main__':
  
    # Calling the function
    camera_list =  [35]#[21,23,24,25,31,32,34,45,41,42,43,44]

    for camera_index in camera_list: 
        print("working on camera: ", camera_index)
        camera_name = "camera{}".format(camera_index)
        save_path = "G:\\PhDProject_real_data\\Brunei_2023_bat_10\\{}\\".format(camera_name)
        try: 
            os.makedirs(save_path)
        except:
            print("Folder exist")
        FrameCapture("G:\\PhDProject_real_data\\Brunei_2023_bat_10\\{}.mp4".format(camera_name), save_path, camera_index) 