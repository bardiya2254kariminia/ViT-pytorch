
import os,sys,argparse
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader ,random_split
from Dataset.datasets import Noisy_Cifar100 , Noisy_STL10
from Dataset.augmentation import Augmentation
from config.path_config import *
from utils.common import compute_total_variation , denormalize_image ,get_visual_map
from config.transform_config  import Transform_class
# losses
import torch.nn.functional as F
from  pytorch_msssim import ssim


class Coach(object):
    def __init__(self , opts):
        self.opts = opts
        self.net = self.get_model().to(self.opts.device)
        self.load_net_weigths()
        if self.opts.mode == "train":
            # optimizers
            self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                betas=(self.opts.b1 , self.opts.b2),
                                lr=self.opts.learning_rate,
                                # weight_decay=self.opts.weigth_decay
                                )
            # Dataloader & Dattset
            self.dataset = self.configurate_dataset()
            self.train_dataset , self.val_dataset = random_split(dataset=self.dataset ,
                                                                  lengths=(self.opts.train_size , len(self.dataset) - self.opts.train_size))
            self.train_dataloader = DataLoader(dataset=self.train_dataset , 
                                         batch_size=self.opts.batch_size,
                                         shuffle=True,
                                         num_workers=self.opts.num_workers,
                                         drop_last=True)
            self.val_dataloader = DataLoader(dataset=self.val_dataset , 
                                         batch_size=self.opts.batch_size,
                                         shuffle=False,
                                         num_workers=self.opts.num_workers,
                                         drop_last=True)
        else:
            self.eval()
    
    def get_model(self):
        if self.opts.model == "U-net":
            return Unet(self.opts)
        elif self.opts.model == "VAE":
            return VAE(self.opts)

    def configurate_dataset(self):
        if self.opts.dataset == "Cifar100":
            return Noisy_Cifar100()
        elif self.opts.dataset == "STL10":
            return Noisy_STL10()

    def load_net_weigths(self):
        try:
            if self.opts.load_weigth_path:
                ckpt = torch.load(self.opts.load_weigth_path, map_location="cpu")
                self.net.load_state_dict(state_dict=ckpt , strict=False)
                print("weigths successfully loaded")
        except:
            print("something went wrong on loading")

    def teach(self):
        total_loss_dict = defaultdict(lambda : [])
        total_loss_val_dict = defaultdict(lambda : [])
        self.net.train()
        for epoch in range(self.opts.epoches) :
            print(f"---------{epoch=}----------")
            # configure the asving methode later
            epoch_loss_dict = defaultdict(lambda : [])
            epoch_val_loss_dict = defaultdict(lambda : [])
            # train_phase
            for i, (clean_image , noisy_image) in tqdm(enumerate(self.train_dataloader , start=1),total= len(self.train_dataloader)):
                clean_image = clean_image.to(self.opts.device)
                noisy_image = noisy_image.to(self.opts.device)
                output_image = self.net(noisy_image)
                # print(f"{output_image=}")
                # sys.exit()
                self.optimizer.zero_grad()
                loss , loss_dict = self.calc_loss(output_image , clean_image)
                # print(f"{loss_dict=}")
                loss.backward()
                self.optimizer.step()
                for key in loss_dict.keys():
                    epoch_loss_dict[key] += loss_dict[key]
            print("train_losses")
            for k , v in epoch_loss_dict.items():
                print(f"{k} = {torch.mean(torch.tensor(v,dtype=torch.float32))}")
                total_loss_dict[k].append(torch.mean(torch.tensor(v,dtype=torch.float32)))
            
            # validation_phase
            # for i, (clean_image , noisy_image) in tqdm(enumerate(self.train_dataloader , start=1),total= (self.train_dataloader)):
            #     clean_image = clean_image.to(self.opts.device)
            #     noisy_image = noisy_image.to(self.opts.device)
            #     self.net.eval()
            #     output_image = self.net(noisy_image)
            #     loss , loss_dict = self.calc_loss(output_image , clean_image)
            #     for key in loss_dict.keys():
            #         epoch_val_loss_dict[key] += loss_dict[key]
            # print("val_losses")
            # for k , v in epoch_val_loss_dict.items():
            #     print(f"{k} = {torch.mean(torch.tensor(v,dtype=torch.float32))}")
            #     total_loss_val_dict[k].append(torch.mean(torch.tensor(v,dtype=torch.float32)))


        # saving weigths
        try:
            print(f"saving weigths in {self.opts.save_weigth_path}")
            print(total_loss_dict)
            self.save_weigths(total_loss_dict , total_loss_val_dict)
        except:
            print("didn't provide saving path")
            sys.exit()

    def calc_loss(self , output_image , target_image):
        loss_dict = defaultdict(lambda:[])
        loss = 0
        if self.opts.model == "U-net":
            # reconstruction loss
            reconstruciton_loss = F.l1_loss(input=output_image , target=target_image)
            loss_dict["reconstruciton_loss"].append(reconstruciton_loss)
            loss +=  self.opts.lambda_reconstruction * reconstruciton_loss
            
            # ssim loss
            ssim_loss = 1- ssim(output_image , target_image)
            loss_dict["ssim_loss"].append(ssim_loss)
            loss +=  self.opts.lambda_ssim * ssim_loss

            # total_variation loss
            tv_loss = compute_total_variation(output_image)
            loss_dict["tv_loss"].append(tv_loss)
            loss +=  self.opts.lambda_total_variation * tv_loss
        elif self.opts.model == "VAE":
            output_image , mean , log_variance = output_image
            # reconstruction loss
            reconstruciton_loss = F.l1_loss(input=output_image , target=target_image)
            loss_dict["reconstruciton_loss"].append(reconstruciton_loss)
            loss +=  self.opts.lambda_reconstruction * reconstruciton_loss
            
            # ssim loss
            ssim_loss = 1- ssim(output_image , target_image)
            loss_dict["ssim_loss"].append(ssim_loss)
            loss +=  self.opts.lambda_ssim * ssim_loss

            # total_variation loss
            tv_loss = compute_total_variation(output_image)
            loss_dict["tv_loss"].append(tv_loss)
            loss +=  self.opts.lambda_total_variation * tv_loss

            # KL Divergence loss
            kl_loss = -0.5 * torch.sum(log_variance - mean**2  - torch.exp(log_variance))
            loss_dict["kl_loss"].append(kl_loss)
            loss +=  self.opts.lambda_kl * kl_loss

        return loss , loss_dict
    
    def test_single(self, image_path):
        input_image = Image.open(image_path)
        input_tensor = Transform_class.get_transform()["cifar100_transform"](input_image).unsqueeze(dim=0)
        output_tensor = self.net(input_tensor)
        denormalize_output = denormalize_image(output_tensor.squeeze(dim=0))
        return Image.fromarray(denormalize_output)
    
    def save_weigths(self , loss_dict, loss_val_dic):
        saving_path = self.opts.save_weigth_path if self.opts.save_weigth_path else os.path.join(os.getcwd() , "Saved_Models")
        os.makedirs(saving_path , exist_ok=True)
        try:
            model_state_dict = self.net.state_dict()
            loss_map = get_visual_map(loss_dict , os.path.join(saving_path , "loss_map.png"))
            # loss_val_map = get_visual_map(loss_val_dic , os.path.join(saving_path , "loss_val_map.png"))
            torch.save(model_state_dict , os.path.join(saving_path , self.opts.model.replace("-" , "_") + "weight.pt"))
        except:
            print(f"failed at saving model and loss-map in {saving_path}")

    def eval(self):
        self.net.eval()