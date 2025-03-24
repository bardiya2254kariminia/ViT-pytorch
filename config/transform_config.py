from torchvision.transforms import transforms
import random
import torch
from PIL import Image
  
# class Add_noise_transform:
#     def __init__(self , noise_std = 0.1):
#         self.noise_types = ["gaussian" , "salt-paper" , "uniform"]
#         # self.noise_types = ["salt-peper-each-channel"]
#         self.noise_std = noise_std

#     def __call__(self,image : torch.Tensor):
#         noise_type = random.choice(self.noise_types)
#         if noise_type == "gaussian":
#             noisy_image = image + torch.rand_like(image) * self.noise_std
#             noise_type = torch.clamp(noisy_image , min=0, max=1)
#             return noisy_image
#         elif noise_type == "salt-peper-each-channel":
#             prob = 0.05
#             noisy_image = torch.rand_like(image)
#             # salt ->1
#             image[noisy_image < prob] = 1
#             # peper ->0
#             image[noisy_image > 1 - prob] = 0
#             return image
#         elif noise_type == "salt-peper":
#             prob = 0.03
#             img = image
#             noise_mask = torch.rand(img.shape[1:], device=img.device)  # Shape: (H, W)
#             # Add salt (value = 1)
#             img[:, noise_mask < prob / 2] = 1.0  

#             # Add pepper (value = 0)
#             img[:, noise_mask > 1 - prob / 2] = 0.0 
#             return img
#         elif noise_type == "uniform":
#             noise = torch.empty_like(image).uniform_(-0.2,0.2)
#             return torch.clamp(image + noise , 0 ,1)

class Add_noise_transform:
    def __init__(self , noise_std = 0.1):
        self.noise_types = ["gaussian" , "salt-peper" , "uniform"]
        self.noise_std = noise_std

    def __call__(self, image: torch.Tensor):
        noise_type = random.choice(self.noise_types)
        image = image.clone()  # Ensure we do not modify the original tensor

        if noise_type == "gaussian":
            noisy_image = image + torch.randn_like(image) * self.noise_std
            return torch.clamp(noisy_image, min=0, max=1)

        elif noise_type == "salt-peper":
            prob = 0.03
            noise_mask = torch.rand(image.shape[1:], device=image.device)  # Shape: (H, W)
            # Add salt (value = 1)
            image[:, noise_mask < prob / 2] = 1.0  
            # Add pepper (value = 0)
            image[:, noise_mask > 1 - prob / 2] = 0.0  
            return image

        elif noise_type == "uniform":
            noise = torch.empty_like(image).uniform_(-0.2, 0.2)
            return torch.clamp(image + noise, 0, 1)

        else:
            # Return original image in case of an undefined noise type
            return image

class Transform_class(object):

    def get_transform():
        t = {
            "cifar100_noisy_transform" : transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                Add_noise_transform(),
                # transforms.Normalize(mean=(0.5,0.5,0.5) , std=(0.5,0.5,0.5))
                transforms.ToPILImage()
            ]),
            "cifar100_transform" : transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5) , std=(0.5,0.5,0.5))
            ])
        }
        return t
    
if __name__ == "__main__":
    tr = Transform_class.get_transform()["cifar100_noisy_transform"]
    img = Image.open(rf"/home/bardiya/projects/ai-side-projects/Deep-Learning-for-Image-Denoising-A-Neural-Restoration-Framework/debugging/img.jpeg")
    out = tr(img)
    out.save("out.jpg")