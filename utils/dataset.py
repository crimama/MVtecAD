import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np 
import cv2

import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms



class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
        self.aug =   transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Resize([256, 256]),
                     transforms.RandomCrop(224),
                     transforms.RandomHorizontalFlip()
                    ])

    def __len__(self):
        return len(self.img_paths)

    def img_load(self,path):
        img = cv2.imread(path)[:,:,::-1]
        img = cv2.resize(img, (256, 256))
        return img 
    
    def img_aug(self,img):
        return self.aug(img)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        img = self.img_load(img)
        if self.mode=='train':
            img = self.img_aug(img)
        if self.mode=='test':
            img = transforms.ToTensor()(img) #<--- ToTensor 과정에서 정규화도 됨 
        
        label = self.labels[idx]
        return img, label