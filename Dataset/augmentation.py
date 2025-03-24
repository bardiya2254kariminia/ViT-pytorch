"""
several augmentation for the datasets
"""
import torch
import torch.nn as nn
from torchvision.transforms import  transforms
from PIL import Image


class Augmentation(object):
    def  __init__(self , opts):
        self.opts = opts

    def get_augmentation(self):
        augment_dict = {
            "denoising_augmentation": transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(128),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.3,hue= 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5) , std=(0.5,0.5,0.5))
            ])
        }
        return augment_dict