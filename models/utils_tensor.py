# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:17:14 2023

@author: 18505
"""



import torch

batch_size = 10
c, h, w = 2, 1024, 1280
nb_points = 472


#img_feat = torch.randn(batch_size, h, w, c).cuda()
x = torch.randint(1, 1024, (10,472, 1))
y = torch.randint(1, 1280, (10,472,1))


x_mask = torch.where(x[:] < 500 and x[:] > 0, torch.tensor(1), torch.tensor(0))
print(x_mask.shape)
y_mask = torch.where(y[:] < 500, torch.tensor(1), torch.tensor(0))
print(y_mask)
'''
xy = (torch.cat([x, y], dim = 2).to(torch.int))
print((xy[:] > torch.tensor([500, 500])).all(dim=2).shape)
mask = (xy[:] > torch.tensor([500, 500])).all(dim=2)
print(mask.shape)
mask = mask.unsqueeze(-1)
mask = mask.repeat(1, 1, 2)
print(mask.shape)
print(mask)
xy = torch.where(mask, xy, torch.tensor([0,0]).to(torch.int))
#print(xy)
# method 1



#result_1 = img_feat[torch.arange(batch_size)[:, None], x.view(batch_size, -1), y.view(batch_size, -1), :]


#print((result_1.shape))
'''