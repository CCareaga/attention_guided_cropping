import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import pandas as pd
import numpy as np

class MelanomaDataset(Dataset):
    def __init__(self, df, imgs, feat, train=True, labels=True, transform=None, chip=False):

        self.df = df
        self.imgs = imgs
        self.feat = feat
        self.train = train
        self.labels = labels
        self.transform = transform
        self.chip = chip
        
    def get_labels(self):
        return list(self.df['target'])
    
    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index])
        img = self.transform(img)

        if self.chip:
            img = img.data.unfold(0, 3, 3).unfold(1, 32, 32).unfold(2, 32, 32)[0]
            img = img.reshape(-1, 3, 32 ,32)

        if self.labels:
            y = self.df.loc[index]['target']
            return img, self.feat[index], torch.ones(1) * y
        else:
            return img, self.feat[index]
    
    def __len__(self):
        return len(self.df)