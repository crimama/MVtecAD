import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import cv2

import os
import timm
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
import time
import matplotlib.pyplot as plt 


device = torch.device('cuda')

from utils.dataset import Custom_dataset
from model.base_model import Network

model = Network()
path = './data/open'
test_png = np.array(sorted(glob(f'{path}/test/*.png')))
train_labels = pd.read_csv("./data/open/train_df.csv")['label']
label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
model.load_state_dict(torch.load('./model/best_model_29.pt'))

#데이터 
test_dataset = Custom_dataset(np.array(test_png),np.array(['tmp']*len(test_png)),mode='test')

test_dataloader = DataLoader(test_dataset,batch_size=8,shuffle=False)

model.eval()
f_pred = [] 

with torch.no_grad():
    for batch in (test_dataloader):
        x = torch.tensor(batch[0],dtype=torch.float32,device=device)
        with torch.cuda.amp.autocast():
            pred = model(x) 
        f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())
label_decoder = {val:key for key, val in label_unique.items()}

f_result = [label_decoder[result] for result in f_pred]
submission = pd.read_csv('./data/open/sample_submission.csv')
submission['label'] = f_result
submission.to_csv('submission0906_1.csv',index=False)

# .py 파일 관련 코드 좀 봐야할 듯 + parser 