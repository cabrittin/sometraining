"""
@name: torch_loader.py                         
@description:                  
    Functions for loading datasets for various pytorch trained models

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import numpy as np
import torch
from torchvision import transforms as T
import cv2

from . import loader

class ObjDetector(torch.utils.data.Dataset):
    def __init__(self,img_paths,**kwargs): 
        """
        img_paths : list of image paths
        """
        self.img_paths = img_paths
        self.img_cache = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        self.img_cache = loader.array_from_stack(self.img_paths[idx])
        self.img_cache = loader.array_16bit_to_8bit(self.img_cache) 
        self.img_cache = loader.image_to_rgb(self.img_cache)
        img = torch.from_numpy(self.img_cache).permute(2,0,1).type(torch.float)
        
        return img / torch.max(img)
    
class Segmenter(torch.utils.data.Dataset):
    def __init__(self, img_paths, transforms=None,dims=None,sample_interval=1):
        self.imgs = np.load(img_paths)
        self.dy = dims[0]
        self.dx = dims[1]
        self.transforms = transforms
        if self.transforms == None:
            self.transforms = T.Compose([T.ToPILImage(),
                            T.Resize((self.dy,self.dx)),
                            T.ToTensor()])

        self.sample_interval = sample_interval

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        if idx % self.sample_interval > 0: return 0 
        
        img = loader.array_16bit_to_8bit(self.imgs[idx,:]).reshape(self.dy,self.dx)
        img = self.transforms(img)
        
        return img

class SegmenterRescale(torch.utils.data.Dataset):
    def __init__(self, img_paths, rescale=0.5,transforms=None,dims=None,sample_interval=1):
        self.imgs = np.load(img_paths)
        self.dy = dims[0]
        self.dx = dims[1]
        
        self.rdy = int(rescale*dims[0])
        self.rdx = int(rescale*dims[1])
 
        self.transforms = transforms
        if self.transforms == None:
            self.transforms = T.Compose([T.ToPILImage(),
                            T.Resize((self.rdy,self.rdx)),
                            T.ToTensor()])

        self.sample_interval = sample_interval

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        if idx % self.sample_interval > 0: return 0 
        
        img = loader.array_16bit_to_8bit(self.imgs[idx,:]).reshape(self.dy,self.dx)
        img = self.transforms(img)
        
        return img


