import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from catalyst.data.sampler import BalanceClassSampler

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

import os
import time
import random
import math
import gc
import argparse

from tqdm import tqdm as tqdm

from resnet import *
from dataset import MelanomaDataset
from util import FocalLoss, DrawHair, gen_train_test_feat

import warnings
warnings.filterwarnings('ignore')

# ResNet50, center crop 224x224, 256d meta layers, embedding dropout 0.8:
# CV: 0.9039
# LB: 0.9203

parser = argparse.ArgumentParser()
parser.add_argument('--folds', type=int, default=3,
                    help='number of folds used for cross-validation')
parser.add_argument('--patience', type=int, default=3,
                    help='number of  epochs of patience until early-stopping')
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to run')
parser.add_argument('--tta', type=int, default=3,
                    help='number of test time augmentation rounds to run')
parser.add_argument('--name', type=str, default='experiment',
                    help='experiment name - used to write out submission file')
parser.add_argument('--resnet', type=str, default='18',
                    help='resnet backbone size to use (18, 34, 50, etc.)')
parser.add_argument('--weights_dir', type=str,
                    help='directory to store model weights')

parser.add_argument('--train_batch_size', type=int, default=64,
                    help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=128,
                    help='validation batch size')
parser.add_argument('--test_batch_size', type=int, default=128,
                    help='testing batch size')

parser.add_argument('--chipped', action='store_true',
                    help='run an experiment with chipped images')
parser.add_argument('--cropped', action='store_true',
                    help='run an experiment with cropped images')
parser.add_argument('--crop_dir', type=str, default='data/crop_data',
                    help='directory holding cropped image data')

parser.add_argument('--attn_drop', type=float, default=0.7,
                    help='dropout rate for attention branch in chipnet')
parser.add_argument('--emb_drop', type=float, default=0.7,
                    help='dropout rate for final embeddings coming from chipnet')
parser.add_argument('--meta_hidden_size', type=int, default=256,
                    help='hidden size of the metadata branch in metanet')
parser.add_argument('--attn_hidden_size', type=int, default=512,
                    help='hidden size of the attention branch in chipnet')

args = parser.parse_args()

# we shouldn't be running an experiment with both chipping and cropping
assert(not (args.chipped and args.cropped))

resnet_dict = {
    '18' : resnet18,
    '34' : resnet34,
    '50' : resnet50,
    '101' : resnet101,
    '152' : resnet152,
}

if args.chipped:
    model_class = ChipNet 
    model_kwargs = {
        'attn_dropout' : args.attn_drop,
        'attn_hidden' : args.attn_hidden_size,
        'emb_dropout' : args.emb_drop,
    }
else:
    model_class = MetaResNet
    model_kwargs = {
        'hidden_sz' : args.meta_hidden_size,
        'dropout' : args.emb_drop
    }

train_transform = transforms.Compose([
    # RGBShift(ranges=[(-30, 50), (-20, 20), (-20, 20)]),
    DrawHair(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(
        brightness=[0.8, 1.2],
        contrast=[0.8, 1.2],
        saturation=[0.8, 1.2],
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    # RGBShift(ranges=[(-20, 20), (-20, 20), (-20, 20)]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(
        brightness=[0.85, 1.15],
        contrast=[0.85, 1.15],
        saturation=[0.85, 1.15],
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

train_csv = pd.read_csv('data/train.csv')
test_csv = pd.read_csv('data/test.csv')

if not args.cropped: 
    train_imgs = np.load('data/train_224.npy')
    test_imgs = np.load('data/test_224.npy')

train_feat, test_feat = gen_train_test_feat(train_csv, test_csv)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=408)

dummy_X = np.zeros(len(train_csv))
train_y = train_csv['target']

# if this isn't a cropped experiment we can create a single test loader
# but in the case of a cropped experiment the test set changes for each fold
if not args.cropped:
    test_dset = MelanomaDataset(
        test_csv, 
        test_imgs, 
        test_feat, 
        train=False, 
        labels=False, 
        transform=test_transform,
        chip=args.chipped
    )
    test_loader = DataLoader(dataset=test_dset, batch_size=args.test_batch_size, shuffle=False)

final_preds = torch.zeros(len(test_csv))
fold_aucs = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X=dummy_X, y=train_y), 1):
    
    print('=' * 20, 'Fold', fold, '=' * 20)

    # if we are using the cropped images, we have to load a different train and
    # test file for each fold to avoid data leakage in our experiments
    if args.cropped:
        train_imgs = np.load(f'{args.crop_dir}/cropped_fold{fold}_train.npy')
        test_imgs = np.load(f'{args.crop_dir}/cropped_fold{fold}_test.npy')

        test_dset = MelanomaDataset(
            test_csv, 
            test_imgs, 
            test_feat, 
            labels=False, 
            transform=test_transform
        )
        test_loader = DataLoader(dataset=test_dset, batch_size=args.test_batch_size, shuffle=False)

    best_model_path = f'{args.weights_dir}/fold{fold}_best.pt'
    final_model_path = f'{args.weights_dir}/fold{fold}_final.pt'
    
    best_auc = 0
    patience = args.patience
    epoch_preds = []
    
    criterion = FocalLoss(logits=True)
    
    resnet = resnet_dict[args.resnet](pretrained=True)
    model = model_class(resnet, **model_kwargs)
    model = model.cuda()
    
    if args.chipped:
        group1 = model.resnet.parameters()
        group2 = list(model.attn_fc1.parameters()) + \
                 list(model.attn_fc2.parameters()) + \
                 list(model.attn_bn1.parameters())
        group3 = list(model.fc4.parameters()) + list(model.bn4.parameters())
            
        optim = torch.optim.Adam(
            [
                {"params": group1, "lr": 5e-5},
                {"params": group2, "lr": 1e-4},
                {"params": group3, "lr": 1e-4},
            ],
        lr=0.00001)

    else:
        group1 = model.resnet.parameters()
        group2 = model.feat_encoder.parameters()

        optim = torch.optim.Adam(
            [
                {"params": group1, "lr": 1e-5},
                {"params": group2, "lr": 5e-5},
            ],
        lr=0.00001)
    
    lmbda = lambda epoch: 0.95
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda=lmbda)
    
    train_df = train_csv.iloc[train_idx].reset_index(drop=True)
    train_sub_imgs = train_imgs[train_idx]
    train_sub_feat = train_feat[train_idx]
    train_dset = MelanomaDataset(
        train_df,  
        train_sub_imgs, 
        train_sub_feat, 
        labels=True, 
        transform=train_transform,
        chip=args.chipped
    )

    valid_df = train_csv.iloc[valid_idx].reset_index(drop=True)
    valid_sub_imgs = train_imgs[valid_idx]
    valid_sub_feat = train_feat[valid_idx]
    valid_dset = MelanomaDataset(
        valid_df, 
        valid_sub_imgs, 
        valid_sub_feat, 
        labels=True,
        transform=valid_transform,
        chip=args.chipped
    )

    train_loader = DataLoader(
        dataset=train_dset, 
        batch_size=args.train_batch_size,  
        sampler=BalanceClassSampler(labels=train_dset.get_labels(), mode="upsampling")
    )
    valid_loader = DataLoader(dataset=valid_dset, batch_size=args.val_batch_size, shuffle=False)
    
    for epoch in range(args.max_epochs):

        correct = 0
        total_loss = 0
        model.train()

        for imgs, feat, labels in tqdm(train_loader):
            
            imgs = imgs.cuda()
            feat = feat.cuda()
            labels = labels.cuda().view(-1)

            optim.zero_grad()

            output = model(imgs, feat)

            logits = output.view(-1)
            loss = criterion(logits, labels)

            loss.backward()
            optim.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        print(f"avg train loss: {train_loss}")
        
        scheduler.step()

        model.eval()

        val_preds, val_labels = [], []
        correct = 0

        for imgs, feat, labels in tqdm(valid_loader):
            with torch.no_grad():
                imgs = imgs.cuda()
                feat = feat.cuda()
                labels = labels.view(-1)

                output = model(imgs, feat)            

                preds = torch.sigmoid(output)
                rounded = torch.round(preds)

                correct += (rounded.cpu().long() == labels.cpu()).sum().item()

                val_preds.extend(list(preds))
                val_labels.extend(list(labels))

        val_preds = np.array(val_preds).reshape(-1)
        val_labels = np.array(val_labels).reshape(-1)

        auc = roc_auc_score(val_labels, val_preds)

        print(f"correct: {correct} out of {len(valid_idx)}")
        valid_acc = correct / len(valid_idx)

        print(f"validation auc: {auc}")
        print(f"validation acc: {valid_acc}")
        
        tta_preds = []
        for i in tqdm(range(args.tta)):
            preds = []
            for imgs, feat in test_loader:
                with torch.no_grad():
                    imgs = imgs.cuda()
                    feat = feat.cuda()

                    output = model(imgs, feat)
                    batch_preds = torch.sigmoid(output).view(-1).cpu()
                    preds.append(batch_preds)

            preds = torch.cat(preds, 0).view(-1)
            tta_preds.append(preds)
        
        epoch_preds.append(torch.stack(tta_preds, 1).mean(1))

        if auc >= best_auc:
            print("saving best weights")
            torch.save(model, best_model_path)

            patience = args.patience
            best_auc = auc
        else:
            patience -= 1
        
        if patience == 0:
            break

    fold_aucs.append(best_auc)

    best_preds = torch.stack(epoch_preds[-args.patience : ], 1).mean(1)
    final_preds += best_preds

    del valid_sub_imgs
    del train_sub_imgs
    gc.collect()

    torch.save(model, final_model_path)

final_preds /= args.folds
sub = pd.read_csv('data/sample_submission.csv')
sub['target'] = final_preds.cpu().numpy().reshape(-1,)
sub.to_csv(f'output/{args.name}_submission.csv', index=False)

print(f"fold best aucs: {fold_aucs}")
oof = sum(fold_aucs) / float(len(fold_aucs))
print(f"avg oof auc: {oof}")






