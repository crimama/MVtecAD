from torch.utils.data import Dataset,DataLoader 
import torch 
import pandas as pd 
import torchvision.transforms as transforms 
import cv2 
import numpy as np 
from PIL import Image 
import os 
from src import augmentation 


def data_factory(cfg):
    
    img_transform, gt_transform = __import__('src').__dict__['augmentation'].__dict__[cfg['DATA']['transform']](384,cfg['DATA']['imgsize'])
    
    trainloader = DataLoader(
                dataset = CustomDataset(
                                        root          = cfg['DATA']['datadir'],
                                        img_size      = cfg['DATA']['imgsize'],
                                        img_transform = img_transform,
                                        gt_transform  = gt_transform,
                                        img_cls       = cfg['DATA']['imgcls'],
                                        mode          = cfg['DATA']['mode'],
                                        train         = True 
                                        ),
                batch_size = cfg['TRAIN']['batchsize'], 
                shuffle    = True
                )
                    
    testloader = DataLoader(
                    dataset = CustomDataset(
                                            root          = cfg['DATA']['datadir'],
                                            img_size      = cfg['DATA']['imgsize'],
                                            img_transform = img_transform,
                                            gt_transform  = gt_transform,
                                            img_cls       = cfg['DATA']['imgcls'],
                                            mode          = cfg['DATA']['mode'],
                                            train         = False
                                            ),
                    #batch_size = cfg['TRAIN']['batchsize'], 
                    batch_size = 1,
                    shuffle    = False
                    )
    return trainloader, testloader 


class CustomDataset(Dataset):
    def __init__(self,root,img_size,img_transform=None,gt_transform=None,img_cls = 'candle',mode='full',train=True):
        super(CustomDataset,self).__init__()
        
        self.root     = root                          # Dataset directory 
        self.dataset  = root.split('/')[-1]
        self.img_size = img_size 
        self.mode     = mode                          # Training mode : Fullshot, 2cls Fewshot, 2cls Highshot 
        self.img_cls  = img_cls                       # Image Class 
        
        self.df = self._read_csv(mode)                # Load df containing information of img and mask 
        self._load_dirs(img_cls,train)                # Following df, load directorys of imgs and masks 
        
        self.img_transform, self.gt_transform =  self._load_transform(img_transform,gt_transform)
        
        self.train = train     
        
        
    def __len__(self):
        return len(self.img_dirs)
            
    def _read_csv(self,mode):
        # Choose a mode of Training : Fullshot, 2cls Fewshot, 2cls Highshot 
        if mode == 'full':
            df = pd.read_csv(os.path.join(self.root,'split_csv','1cls.csv'))
        elif mode == 'fewshot':
            df = pd.read_csv(os.path.join(self.root,'split_csv','2cls_fewshot.csv'))
        elif mode == 'highshot':
            df = pd.read_csv(os.path.join(self.root,'split_csv','2cls_highshot.csv'))
        return df 
    
    def _load_dirs(self,img_cls,train):
        # Choose either Training or Test and additionaly Class of img (ex : Candle)
        if img_cls == 'all': # In case using All type of Image Claases 
            if train:
                self.img_dirs = self.df[self.df['split']=='train']['image'].values
                self.gt_dirs = self.df[self.df['split']=='train']['mask'].values
            else:
                self.img_dirs = self.df[self.df['split']=='test']['image'].values
                self.gt_dirs = self.df[self.df['split']=='test']['mask'].values
        else: # In case only using one class of image 
            if train:
                self.img_dirs = self.df[(self.df['split']=='train') & (self.df['object'] == self.img_cls)]['image'].values
                self.gt_dirs = self.df[(self.df['split']=='train') & (self.df['object'] == self.img_cls)]['mask'].values
            else:
                self.img_dirs = self.df[(self.df['split']=='test') & (self.df['object'] == self.img_cls)]['image'].values
                self.gt_dirs = self.df[(self.df['split']=='test') & (self.df['object'] == self.img_cls)]['mask'].values
                
    def _load_transform(self,img_transform,gt_transform):
        if img_transform == None:
            img_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((self.img_size,self.img_size))
                                ])
        if gt_transform == None:
            gt_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((self.img_size,self.img_size))
                                ])
        return img_transform, gt_transform
    
    def load_img(self,img_dir):
        img = Image.open(os.path.join(self.root,img_dir)).convert('RGB')
        img = self.img_transform(img)        
        return img 

    def load_gt(self,gt_dir):
        try:
            gt = Image.open(os.path.join(self.root,gt_dir))
            if self.dataset == 'ViSA':
                gt = np.array(gt)*255
            #gt = cv2.resize(np.array(gt),dsize=(self.img_size,self.img_size))
            #gt = np.expand_dims(gt,axis=-1)
            gt = self.gt_transform(gt)
        except:
            gt = np.zeros((self.img_size,self.img_size,1))
        return gt 
        
    
    def __getitem__(self,idx):
        img_dir = self.img_dirs[idx]
        gt_dir = self.gt_dirs[idx]
        
        img = self.load_img(img_dir)
        gt = self.load_gt(gt_dir)
        
        if np.sum(gt) == 0:
            label = torch.Tensor([0])
        else:
            label = torch.Tensor([1])
        
        return img,gt,label