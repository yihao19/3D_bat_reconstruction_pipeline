# -*- coding: utf-8 -*-
"""
attention mechanism with VGG 16 feature

@author: 18505
"""
import torch
import torch.nn as nn
import cv2 as cv
import torchvision.models as models





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True).cuda()



test_image_path = "G:/GCN_project/Bat_Sample_Model/high_speed_tunnel/rearranged/pose242/camera19.png"
image = cv.imread(test_image_path)
image = cv.resize(image, (244, 244)).astype('float32') / 255.
image = image.transpose(2, 0, 1)

model = torch.nn.Sequential(*(list(model.children())[:-2]))
image = torch.tensor(image).cuda()
image = image.unsqueeze(0)

print(model)
print(image.shape)

output = model(image)
output = torch.flatten(output, 1)
print(output.shape)

# inference