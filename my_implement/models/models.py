import torch
import torch.nn as nn
import copy
import math
import os,sys
from torch.nn import CrossEntropyLoss, Dropout,Softmax,Linear,Conv2d,LayerNorm
from torch.nn.modules.utils import _pair


"""
implementation of ViT Based on the paper:
    AN IMAGE IS WORTH 16 X 16 WORDS :
    TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
    link: https://arxiv.org/abs/2010.11929v2

reimplementation by bardiya2254kariminia
"""

class Embedding(nn.Module):
    def __init__(self, opts, in_channels = 3):
        super(Embedding,self).__init__()
        self.opts = opts
        self.in_channels = in_channels
        self.hybrid_model = None
        image_size  = _pair(self.opts.image_size)
        if self.opts.hybrid_model:
            grid_size = _pair(self.opts.grid_size)
            patch_size = (image_size[0] // self.opts.patch_size // grid_size[0] ,image_size[1] // self.opts.patch_size//grid_size[1])
            patch_number = (image_size[0] // self.opts.patch_size) * (image_size[1] // self.opts.patch_size)
            self.hybrid_model = True
        else:
            patch_size = _pair(self.opts.patch_size)
            patch_number = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
            self.hybrid_model = False

        if self.hybrid_model:
            self.hybrid_backbone = None    # should fix this later
            in_channels =  None  # should fix later
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels= self.opts.hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.positional_embedding = nn.Parameter(torch.ones(1,patch_number+1 , self.opts.hidden_size))
        for i in range(patch_number+1):
            self.positional_embedding[:,i,:] = self.positional_embedding[:,i,:] * i
        self.cls_token = nn.Parameter(torch.ones(1,1, self.opts.hidden_size))
        self.dropout = Dropout(p=0.2)

    def forward(self,x:torch.Tensor):
        b , c, h  ,w = x.shape
        cls_token = self.cls_token.repeat((b,-1,-1))

        if self.hybrid_model:
            x = self.hybrid_backbone(x)
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x.transpose(-1,-2)
        x = torch.cat([cls_token,x],dim=1)
        out= x + self.positional_embedding
        out = self.dropout(out)
        return out

class MLP(nn.Module):
    def __init__(self,opts):
        super(MLP, self).__init__()
        self.opts = opts
        self.fc1 = nn.Linear(in_features=self.opts.hidden_size , out_features=self.opts.hidden_size*4)
        self.fc2 = nn.Linear(in_features=self.opts.hidden_size , out_features=self.opts.hidden_size*4)
        self.dropout = nn.Dropout(p=0.2)
        self.gelu = nn.GELU()
        self.body =nn.Sequential(
            self.fc1,
            self.gelu,
            self.dropout,
            self.fc2,
            self.dropout
        )

    def __init_weigths(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias ,std=1e-6)
        nn.init.normal_(self.fc2.bias ,std=1e-6)

    def forward(self,x:torch.Tensor):
        return self.body(x)
    
class Attention_Block(nn.Module):
    def __init__(self,opts):
        super(Attention_Block,self).__init__()
        pass

    
    def forward(x:torch.Tensor):
        pass

class Transformer_Block(nn.Module):
    def __init__(self, opts):
        super(Transformer_Block,self).__init__()
        self.opts = opts
        self.mlp = MLP(opts=self.opts)
        self.layer_norm1 = LayerNorm(self.opts.hidden_size ,eps=1e-7)
        self.layer_norm2 = LayerNorm(self.opts.hidden_size ,eps=1e-7)
        self.attention_block = None


    def forward(self, x:torch.Tensor):
        b,c,h,w = x.shape
        x1 = self.layer_norm1(x)
        x2, att_weigths = self.attention_block(x1)
        x3 = x2 + x1

        x3 = self.layer_norm2(x3)
        x4 = self.mlp(x3)
        x4 = x4 + x3
        return x3,att_weigths