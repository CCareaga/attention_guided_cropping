import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import pandas as pd
import numpy as np

import cv2
from PIL import Image, ImageDraw, ImageFilter

from skimage.measure import regionprops


from tqdm import tqdm as tqdm

hw = 224

def get_square_bbox(bb):
    diffs = (bb[3] - bb[1], bb[2] - bb[0])

    center = (bb[1] + (diffs[0] // 2), bb[0] + (diffs[1] // 2))
    short_len = max(*diffs)
    half_short_len = short_len // 2
    return [
        np.clip(center[0] - half_short_len, 0, 224), 
        np.clip(center[1] - half_short_len, 0, 224), 
        np.clip(center[0] + half_short_len, 0, 224), 
        np.clip(center[1] + half_short_len, 0, 224)
    ]

def zoom_enhance(model_path, csv_path, data_path, full_sz_path, output_path, num_devs=1.5):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    model = torch.load(model_path)
    model.eval()
    csv = pd.read_csv(csv_path)
    data = np.load(data_path)
    
    images = []
    for img, fname in tqdm(zip(data, csv['image_name'])):
        
        np_img = img.copy()
        img = Image.fromarray(img)

        img = transform(img)
        img = img.cuda()

        chips = img.data.unfold(0, 3, 3).unfold(1, 32, 32).unfold(2, 32, 32)[0]
        chips = chips.reshape(-1, 3, 32, 32)

        output = model(chips.unsqueeze(0), None)

        attn_map = model.attn_map[0]

        upscaled = cv2.resize(attn_map.view(7, 7).detach().cpu().numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)
        binary = (upscaled > (upscaled.mean() + upscaled.std() * num_devs)).astype(np.uint8)

        props = regionprops(binary)
        prop = props[0]
        sqr_bb = get_square_bbox(prop.bbox)
        
        full_sz = Image.open(f'{full_sz_path}{fname}.jpg')
        full_hw = float(full_sz.size[0])

        adj_bb = [pt * (full_hw / hw) for pt in sqr_bb]

        cropped = transforms.functional.crop(
            full_sz, 
            adj_bb[1], 
            adj_bb[0], 
            adj_bb[2] - adj_bb[0], 
            adj_bb[3] - adj_bb[1]
        )
        cropped = transforms.functional.resize(cropped, (224, 224), interpolation=5)
        cropped_arr = np.array(cropped)
        
        images.append(cropped_arr)
    
    output = np.stack(images, axis=0)
    np.save(output_path, output)


zoom_enhance(
    'weights/chip_weights/fold1_final.pt', 
    'data/train.csv', 
    'data/train_224.npy',
    'data/fullsize/train/',
    'data/crop_data/fold1_train.npy'
)

zoom_enhance(
    'weights/chip_weights/fold1_final.pt', 
    'data/test.csv', 
    'data/test_224.npy',
    'data/fullsize/test/',
    'data/crop_data/fold1_test.npy'
)

zoom_enhance(
    'weights/chip_weights/fold2_final.pt', 
    'data/train.csv', 
    'data/train_224.npy',
    'data/fullsize/train/',
    'data/crop_data/fold2_train.npy'
)

zoom_enhance(
    'weights/chip_weights/fold2_final.pt', 
    'data/test.csv', 
    'data/test_224.npy',
    'data/fullsize/test/',
    'data/crop_data/fold2_test.npy'
)

zoom_enhance(
    'weights/chip_weights/fold3_final.pt', 
    'data/train.csv', 
    'data/train_224.npy',
    'data/fullsize/train/',
    'data/crop_data/cropped_fold3_train.npy'
)

zoom_enhance(
    'weights/chip_weights/fold3_final.pt', 
    'data/test.csv', 
    'data/test_224.npy',
    'data/fullsize/test/',
    'data/crop_data/cropped_fold3_test.npy'
)