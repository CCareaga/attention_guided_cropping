import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import pandas as pd
import numpy as np

class MelanomaDataset(Dataset):
    def __init__(self, df, imgs, feat, train=True, labels=True, transform=None, chip=False):
        # this class is used for all models, so it has functionality for chipping
        self.df = df
        self.imgs = imgs
        self.feat = feat
        self.train = train
        self.labels = labels
        self.transform = transform
        self.chip = chip
        
    def get_labels(self):
        # return the labels for this dataset, used in the training code
        return list(self.df['target'])
    
    def __getitem__(self, index):
        # index the numpy array of data, and convert to PIL Image
        img = Image.fromarray(self.imgs[index])

        # run the specified transformation (augmentation and normalization)
        img = self.transform(img)

        # if chipping is on use unfold to create 32 x 32 tiles
        if self.chip:
            img = img.data.unfold(0, 3, 3).unfold(1, 32, 32).unfold(2, 32, 32)[0]
            img = img.reshape(-1, 3, 32 ,32)

        # if the dataset has labels (train or validation) then return them along
        # with the input image and features
        if self.labels:
            y = self.df.loc[index]['target']
            return img, self.feat[index], torch.ones(1) * y
        else:
            return img, self.feat[index]
    
    def __len__(self):
        return len(self.df)