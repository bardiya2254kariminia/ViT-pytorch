import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__) , "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__) , "../..")))
import torch
import torch.nn as nn
import copy
import math
from torch.nn import CrossEntropyLoss, Dropout,Softmax,Linear,Conv2d,LayerNorm
from torch.nn.modules.utils import _pair
import json
from argparse import Namespace

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

        self.positional_embedding = nn.Parameter(torch.ones(1,patch_number+1 , self.opts.hidden_size),requires_grad=True)
        with torch.no_grad():
            for i in range(patch_number+1):
                self.positional_embedding[:, i, :] = self.positional_embedding[:, i, :] * i
        self.cls_token = nn.Parameter(torch.ones(1,1, self.opts.hidden_size),requires_grad=True)
        self.dropout = Dropout(p=0.2)

    def forward(self,x:torch.Tensor):
        if self.hybrid_model:
            x = self.hybrid_backbone(x)
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(-1,-2)
        x = torch.cat([self.cls_token,x],dim=1)
        out= x + self.positional_embedding
        out = self.dropout(out)
        return out

class MLP(nn.Module):
    def __init__(self,opts):
        super(MLP, self).__init__()
        self.opts = opts
        self.fc1 = nn.Linear(in_features=self.opts.hidden_size, out_features=self.opts.hidden_size*4)
        self.fc2 = nn.Linear(in_features=self.opts.hidden_size * 4 , out_features=self.opts.hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.gelu = nn.GELU()
        self.body =nn.Sequential(
            self.fc1,
            self.gelu,
            self.dropout,
            self.fc2,
            self.dropout
        )
        self.__init_weigths()

    def __init_weigths(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias ,std=1e-6)
        nn.init.normal_(self.fc2.bias ,std=1e-6)

    def forward(self,x:torch.Tensor):
        # print(f"{x.shape=}")
        return self.body(x)
    
class Attention(nn.Module):
    def __init__(self,opts):
        super(Attention,self).__init__()
        self.opts = opts
        self.num_heads = self.opts.num_attention_heads
        self.head_dim = self.opts.hidden_size / self.num_heads
        self.head_scale = math.sqrt(self.head_dim)
        
        # for case that the input embeds are the same for the qkv.
        self.Q_proj = nn.Linear(in_features=self.opts.hidden_size, 
                                out_features=self.opts.hidden_size)
        self.K_proj = nn.Linear(in_features=self.opts.hidden_size,
                                out_features=self.opts.hidden_size)
        self.V_proj = nn.Linear(in_features=self.opts.hidden_size, 
                                out_features=self.opts.hidden_size)

        self.out = nn.Linear(in_features=self.opts.hidden_size , out_features=self.opts.hidden_size)
        self.projection_dropout = nn.Dropout(p=0.1)
        self.attention_dropout = nn.Dropout(p=0.1)

    def forward(self,x:torch.Tensor):
        batch , seq_len , embed_dim = x.shape
        # print(f"{x.shape=}")
        # getting the qkv
        q = self.projection_dropout(self.Q_proj(x))
        k = self.projection_dropout(self.K_proj(x))
        v = self.projection_dropout(self.V_proj(x))
        # resize for the mha
        q = q.reshape(batch , self.num_heads , seq_len , -1)
        k = k.reshape(batch , self.num_heads , seq_len , -1)
        v = v.reshape(batch , self.num_heads , seq_len , -1)
        # print(f"{q.shape=} {k.shape=}{v.shape=} ")
        # compute the attention scores
        attn_score = (q @ k.transpose(-1,-2))/self.head_scale  #(batch , num_heads ,seq_len , seq_len)
        attn_weigths = torch.softmax(attn_score , dim=-1) #(batch , num_heads , seq_len , seq_len)
        # print(f"{attn_weigths.shape=}")
        # attentoin outputs
        outputs = attn_weigths @ v 
        # print(f"{outputs.shape=}")
        outputs = outputs.reshape(batch , seq_len, embed_dim)
        outputs = self.attention_dropout(self.out(outputs))
        return outputs , attn_weigths
    
class Transformer_Block(nn.Module):
    def __init__(self, opts):
        super(Transformer_Block,self).__init__()
        self.opts = opts
        self.mlp = MLP(opts=self.opts)
        self.layer_norm1 = LayerNorm(self.opts.hidden_size ,eps=1e-7)
        self.layer_norm2 = LayerNorm(self.opts.hidden_size ,eps=1e-7)
        self.attention_block = Attention(opts = self.opts)

    def forward(self, x:torch.Tensor):
        x1 = self.layer_norm1(x)
        x2, att_weigths = self.attention_block(x1)
        x3 = x2 + x1

        x3 = self.layer_norm2(x3)
        x4 = self.mlp(x3)
        x4 = x4 + x3
        return x3,att_weigths

class Transformer_Encoder(nn.Module):
    def __init__(self, opts):
        super(Transformer_Encoder, self).__init__()
        self.opts = opts
        # Notice: a nn.Linear layer only consider the last layer if the  tensor has more than 2 dims
        self.norm_layer = nn.LayerNorm(self.opts.hidden_size  ,eps=1e-7)
        self.layer_list = nn.ModuleList()
        for i in range(self.opts.num_transformer_layers):
            self.layer_list.add_module(
                f"Transformer_Block_{i}" , Transformer_Block(self.opts)
            )
    def forward(self , x:torch.Tensor):
        att_weigths_list = []
        for layer in self.layer_list:
            x , att_weights = layer(x)
            att_weigths_list.append(att_weights)
        # last layer norm for the mlp
        x = self.norm_layer(x)
        return x , att_weigths_list
        
class Transformer(nn.Module):
    def __init__(self,opts):
        super(Transformer ,self).__init__()
        self.opts = opts
        self.embedd_layer = Embedding(opts=self.opts)
        self.transformer_encoder= Transformer_Encoder(opts=self.opts)
    
    def forward(self, x:torch.Tensor):
        x =self.embedd_layer(x)
        out , att_weigth_list = self.transformer_encoder(x)
        return out, att_weigth_list

class  Vision_Transformer(nn.Module):
    def __init__(self, opts):
        # super(Vision_Transformer , self).__init__()
        super().__init__()
        self.opts = opts
        self.transformer= Transformer(opts = self.opts)
        self.mlp_head = nn.Linear(in_features=self.opts.hidden_size , out_features= self.opts.num_class)

    def forward(self,x , return_attention_weigths = False,  return_logits= False):
        out , att_weigths = self.transformer(x)
        z0_l = out[:,0]
        output = self.mlp_head(z0_l)
        if return_attention_weigths:
            return output if return_logits else torch.softmax(output, dim=-1), att_weigths
        return output if return_logits else torch.softmax(output, dim=-1)


if __name__ == "__main__":
    with open(rf"/home/bardiya/projects/ai-side-projects/ViT-pytorch/config.json" , "r") as f:
        opts = json.load(f)
        opts = Namespace(**opts)
    model = Vision_Transformer(opts = opts)
    # print(model)
    out= model(torch.zeros(1,3,224,224))
    print(out.shape)