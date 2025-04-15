"""
implementation of several  dataset classes for different models
presented by bardiya2254kariminia@github.com

now we cannot train them on large datasets 
so we are going to use the cifar 
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100 , STL10 , CIFAR10
from config.transform_config import Transform_class

# implementation of datasets
class Cifar100(Dataset):
    def __init__(self):
        super(Cifar100, self).__init__()
        self.cifar = CIFAR100(root="./data", train=True, download=True)

    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, index):
        image , label = self.cifar[index]
        image = Transform_class.get_transform()["cifar100_transform"](image)
        return image , label
    
class Cifar10(Dataset):
    def __init__(self):
        super(Cifar10, self).__init__()
        self.cifar = CIFAR10(root="./data", train=True, download=True)

    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, index):
        image , label = self.cifar[index]
        image = Transform_class.get_transform()["cifar100_transform"](image)
        return image , label

class STL10(Dataset):
    def __init__(self ):
        super(STL10,self).__init__()
        self.cifar = STL10(root="./data", split="test", download=True)

    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, index):
        image , label = self.cifar[index]
        clean_image = Transform_class.get_transform()["cifar100_transform"](image)
        return clean_image , label
    

if __name__ == "__main__":
    ds =  Cifar100()
    print(ds.__len__())