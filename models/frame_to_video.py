# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:52:31 2023

@author: 18505
"""

import cv2
import numpy as np
import glob

frameSize = (720, 720)

camera = "bat_test_13_1_high_speed"
out = cv2.VideoWriter('{}.mp4'.format(camera),cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)

for filename in glob.glob('D:/PhDProject_real_data/render_image_2023_bat_test_13_1/*.png'):
    img = cv2.imread(filename)
    out.write(img)

out.release()