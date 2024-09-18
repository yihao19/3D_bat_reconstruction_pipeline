    n# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:33:40 2023

@author: 18505

define the attention mechanism 
"""


import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads = 1):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embeded size needs to be div by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # output the same size
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        # value/keys/query shape: (1, #camera_number, # flatten_silouette_pixels)
        # batch size is going to be 1
        # value_len, key_len, query_len = # camera_number
        
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # plist embedding into self.heads pieces
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)
        
        
        # shape: (1, # camera_number, 1, # flatten_silouette_pixel)
        # shape: (1, # camera_number, 1, # flatten_silouette_pixel)
        # shape: (1, # camera_number, 1, # flatten_silouette_pixel)
        values = self.values(values)
        keys = self.keys(keys)
        querys = self.queries(queries)
        
        # what is this statement
        # queries: 1, # camera_number, 1, # flatten_silouette_pixel
        # keys   : 1, # camera_number, 1, # flatten_silouette_pixel
        # output : 1, 1, queries, key (1, 1, 10, 10)
        # interaction between the 10 camera information
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        '''
        if mask != None:
            energy = energy.masked_fill(mask == 0, float("-1e28"))
        '''  
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) 
        # attention shape = (N, heads, query_len, key_len)
        # values shape = (N, value_len, heads, heads_dim)
        # output = (N, query_len, heads, heads_dim)
        # attention shape: (1, 1, 10, 10)
        # values shape: 1, 10, 1, # flatten_silouette_pixel
        # output shape: 1, 10, 1, # flatten_silouette_pixel
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim)
        # output shape: 1, 10, flatten_silouette_pixel
        out = self.fc_out(out)
        # output for the second layer
        return out
'''
define the transformation block including the layer normalization and residule block
'''
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size), 
            nn.ReLU(), 
            nn.Linear(forward_expansion*embed_size, embed_size)
            )
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out
'''
define the transformer encoder to produce the weight for each image
'''
class Encoder(nn.Module):
    def __init__(
            self,
            input_size,
            num_data, 
            embed_size, 
            num_layers, 
            heads, 
            device, 
            forward_expansion, 
            dropout, 
            ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.num_data = num_data
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, 
                                 heads, 
                                 dropout=dropout, 
                                 forward_expansion=forward_expansion
                                 )
                for _ in range(num_layers)
                ]
            )
        self.flatten = nn.Flatten(start_dim = 1)
        self.fc = nn.Linear(self.embed_size * self.num_data, self.num_data)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, embed_size)
        self.softmax = nn.Tanh()
    def forward(self, x, mask):
        out = self.linear(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)
    
        
        out = self.softmax(out)
        return out
if __name__=="__main__":
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 3
    # input (# pose, joint_matrix) shape should be (#pose, 23, 3)
    # output shape is the same as the input shape
    src = torch.rand((10, 23, 3)).to(device)
    trg = torch.rand((10, 23, 3)).to(device)
    # target is also the torch.rand((1, 23, 3))
    embed_size = 32
    model = Encoder(image_size, 10, 3, 8, 1, device, 1, 0).to(device)
    
    out = model(src, None)
    print(out.shape)
        
        
