import torch
import numpy as np
from PIL import Image
import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt

def denormalize_image(img:torch.Tensor , mean = (0.5,0.5,0.5) , std = (0.5,0.5,0.5)):
    img = img.cpu().detach().transpose(0,2).transpose(0,1).numpy()
    mean = np.array(mean).reshape(1,1,3)
    std = np.array(std).reshape(1,1,3)

    img  = img * std + mean
    img = np.clip(img , 0,1)
    img = (img * 255).astype("uint8")
    return img

def load_json(json_path:str):
    with open(json_path ,"r") as f:
        data = json.load(f)
    return data

def compute_total_variation(image : torch.Tensor):  # image = (batch,C,H,W)
    H_smoothing = torch.abs(image[:,:,:-1,:] - image[:,:,1:,:])
    W_smoothing = torch.abs(image[:,:,:,:-1] - image[:,:,:,1:])
    return (torch.mean(H_smoothing) + torch.mean(W_smoothing))/ (image.shape[-1]**2)

def get_visual_map(loss_dict, save_path):
    colors = plt.cm.viridis(np.linspace(0 , 1, len(loss_dict)))
    plt.figure(figsize=(8,6))
    for i , (k , v) in enumerate(loss_dict.items()):
        plt.plot(v ,label= k , color= colors[i])
    plt.xlabel("Epoche")
    plt.ylabel("loss_value")
    plt.title("loss_map")
    plt.legend()
    plt.savefig(save_path)
    