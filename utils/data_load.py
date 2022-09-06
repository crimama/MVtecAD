from glob import glob 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
import os 
import random 

def data_load(path):

    
    train_y = pd.read_csv("./data/open/train_df.csv")

    train_labels = train_y["label"]

    label_unique = sorted(np.unique(train_labels))
    label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

    train_labels = [label_unique[k] for k in train_labels]
    
    return train_png, test_png, train_labels, label_unique 


class data_path_load:
    def __init__(self,path,shuffle=True,validation_split=True):
        self.train_png = sorted(glob(f'{path}/train/*.png'))
        self.test_png = sorted(glob(f'{path}/test/*.png'))
        self.train_df = pd.read_csv(f'{path}/train_df.csv')
        self.train_labels = self.train_df['label']
        self.label_unique = self.make_label_unique() 
        self.train_labels = [self.label_unique[k] for k in self.train_labels]
        self.validation_split = validation_split
        self.shuffle = shuffle 
        
    def make_label_unique(self):
        label_unique = sorted(np.unique(self.train_labels))
        label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
        return label_unique 
        
    def data_shuffle(self):
        shuffle_idx = np.arange(len(self.train_png))
        np.random.shuffle(shuffle_idx)
        self.train_png = np.array(self.train_png)[shuffle_idx]
        self.train_labels = np.array(self.train_labels)[shuffle_idx]
        
        return self.train_png, self.train_labels 
    
    def validation_split(self,train_png,train_labels):
        split_idx = int(len(train_png)*0.2)
        valid_png = train_png[:split_idx]
        train_png = train_png[split_idx:]

        valid_labels = train_labels[:split_idx]
        train_labels = train_labels[split_idx:]
        return [train_png, train_labels, valid_png, valid_labels]

    def __call__(self):
        if self.validation_split==False:
            if self.shuffle:
                self.train_png, self.train_labels = self.data_shuffle()
            return self.train_png, self.train_labels

        if self.validation_split:
            if self.shuffle:
                self.train_png, self.train_labels = self.data_shuffle()
            data = self.validation_split(self.train_png,self.train_labels)
            [train_png, train_labels, valid_png, valid_labels] = data
            
            return train_png, train_labels, valid_png, valid_labels
            
    
        
        
        