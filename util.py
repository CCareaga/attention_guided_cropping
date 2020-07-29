import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import numpy as np
import pandas as pd
import random
import cv2
from PIL import Image, ImageDraw, ImageFilter

def gen_train_test_feat(train_df, test_df):
    combined_sex = train_df['sex'].values.tolist() + test_df['sex'].values.tolist()
    combined_age = train_df['age_approx'].values.tolist() + test_df['age_approx'].values.tolist()
    combined_site = train_df['anatom_site_general_challenge'].values.tolist() + test_df['anatom_site_general_challenge'].values.tolist()

    combined_sex = pd.get_dummies(combined_sex, dummy_na=True).values.tolist()
    combined_age = pd.get_dummies(combined_age, dummy_na=True).values.tolist()
    combined_site = pd.get_dummies(combined_site, dummy_na=True).values.tolist()

    sex_meta = torch.Tensor(combined_sex)
    age_meta = torch.Tensor(combined_age)
    site_meta = torch.Tensor(combined_site)

    meta_feat = torch.cat((sex_meta, age_meta, site_meta), 1)

    return meta_feat[: len(train_df), :], meta_feat[len(train_df) :, :]

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DrawHair:
    def __init__(self, count=1000, size=224):

        self.size = size
        self.colors = [
            (35, 18, 11),
            (61, 35, 20),
            (90, 56, 37),
            (44, 22, 8),
            (21, 15, 8),
        ]
        
        self.premade_hairs = [self.generate_hairs() for _ in range(count)]
        
    def generate_hairs(self):
        num_hairs = random.choice([0, 0, 50, 100, 150, 200, 300, 400, 500, 600])
        color = list(random.choice(self.colors))

        width, height = (self.size + 200, self.size + 200)
        
        hair_img = Image.new('RGBA', (width, height), tuple(color + [0]))
        draw = ImageDraw.Draw(hair_img)
        
        for i in range(num_hairs):
            direction = random.choice([[1, 1], [-1, 1], [1, -1]])
            hair_color = tuple(color + [random.randint(70, 180)])
            origin = random.randint(-width, width), random.randint(-height, height)

            # choose a random direction for the hair
            end0 = random.randint(100, width * 1.5) * direction[0]
            end1 = random.randint(100, height * 1.5) * direction[1]

            end = (origin[0] + end0, origin[1] + end1)

            angles = (-random.randint(0, 70), random.randint(0, 70))
            draw.arc([origin, end], angles[0], angles[1], fill=hair_color, width=random.choice([1, 2, 3]))
        
        hair_img = hair_img.filter(ImageFilter.GaussianBlur(radius=1))
        return hair_img.resize((self.size, self.size), resample=Image.LANCZOS)

    def __call__(self, img):

        img = img.convert('RGBA')
        random_hairs = random.choice(self.premade_hairs)
        
        out = Image.alpha_composite(img, random_hairs).convert('RGB')
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'

class DownUp:
    def __init__(self, scales=[0.7, 0.6, 0.5], p=0.5):

        self.p = p
        self.scales = scales

    def __call__(self, img):
        if random.random() < self.p:
            h, w = img.size
            scale = random.choice(self.scales)
            down_h, down_w = int(h * scale), int(w * scale)
            down = img.resize((down_h, down_w), Image.LANCZOS)
            return down.resize((h, w), Image.BILINEAR)
            
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}'

    