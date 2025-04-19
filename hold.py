import torch

a = torch.zeros((1,3,224,224))
a.repeat((8,1,1))