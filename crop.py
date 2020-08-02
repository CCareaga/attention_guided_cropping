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
    # this function takes a bounding box of the form:
    # [y1, x1, y2, x2] and converts it to a square bounding box with the
    # same center with a side length matching the long side of the original bb

    # compute change in x and change in y (side lengths)
    diffs = (bb[3] - bb[1], bb[2] - bb[0])

    # determine the center of the input bounding box
    center = (bb[1] + (diffs[0] // 2), bb[0] + (diffs[1] // 2))

    # determine the length of the long side
    short_len = max(*diffs)

    # divide shot side by two to determine how far the to move from center
    half_short_len = short_len // 2

    # return a square bounding box by offsetting from the center of the bb
    # make sure to not go past 0 or 224 to stay in the image
    return [
        np.clip(center[0] - half_short_len, 0, 224), 
        np.clip(center[1] - half_short_len, 0, 224), 
        np.clip(center[0] + half_short_len, 0, 224), 
        np.clip(center[1] + half_short_len, 0, 224)
    ]

def zoom_enhance(model_path, csv_path, data_path, full_sz_path, output_path, num_devs=1.5):
    
    # just do imagenet normalization and nothing else
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    # load the model, csv with metadata and numpy data array
    model = torch.load(model_path)
    model.eval()
    csv = pd.read_csv(csv_path)
    data = np.load(data_path)
    
    # for each image in the numpy array of input data, and the names 
    # of the original images from the csv file
    images = []
    for img, fname in tqdm(zip(data, csv['image_name'])):
        
        # conver the image to a PIL Image and normalize it
        img = Image.fromarray(img)
        img = transform(img)
        img = img.cuda()

        # perform chipping operation that the chipnet model expects
        chips = img.data.unfold(0, 3, 3).unfold(1, 32, 32).unfold(2, 32, 32)[0]
        chips = chips.reshape(-1, 3, 32, 32)

        # send the chips through the model (output doesn't matter here)
        output = model(chips.unsqueeze(0), None)

        # we only sent through a single image so grab the zeroth item in the
        # attention map that gets stored in the model when data is passed
        attn_map = model.attn_map[0]

        # bilinearly upscale the attention map to match the input image size
        # I'm not exactly sure why I used cv for this rather than PIL
        upscaled = cv2.resize(attn_map.view(7, 7).detach().cpu().numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)

        # threshold the image using the num_devs parameter to generate a binary mask
        binary = (upscaled > (upscaled.mean() + upscaled.std() * num_devs)).astype(np.uint8)

        # use skimage to compute the minimum bounding box that encapsulates all pixels in the 
        # binary mask with value of one, then convert result to a square
        props = regionprops(binary)
        prop = props[0]
        sqr_bb = get_square_bbox(prop.bbox)
        
        # open the original high resolution image and determine it's size
        full_sz = Image.open(f'{full_sz_path}/{fname}.jpg')
        full_hw = float(full_sz.size[0])

        # adjust the bounding box from the small image to match the
        # corresponding area in the high resolution image
        adj_bb = [pt * (full_hw / hw) for pt in sqr_bb]

        # crop the original image using the adjusted bounding box
        cropped = transforms.functional.crop(
            full_sz, 
            adj_bb[1], 
            adj_bb[0], 
            adj_bb[2] - adj_bb[0], 
            adj_bb[3] - adj_bb[1]
        )
        
        # resize the cropped image to 224 x 224 and convert to an array
        cropped = transforms.functional.resize(cropped, (224, 224), interpolation=5)
        cropped_arr = np.array(cropped)
        
        images.append(cropped_arr)
    
    # stack all the 224 x 224 image and store in output path
    output = np.stack(images, axis=0)
    np.save(output_path, output)

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='',
                    help='path to model weights for chipnet')

parser.add_argument('--csv_path', type=str, default='',
                    help='path to csv with metadata for input data')

parser.add_argument('--data_path', type=str, default='',
                    help='path to the numpy array containing the dataset')

parser.add_argument('--full_sz_path', type=str, default='',
                    help='path to full size images (referenced in each row of csv')

parser.add_argument('--output_path', type=str, default='',
                    help='path to store the resulting cropped dataset')

parser.add_argument('--num_devs', type=float, default=1.5,
                    help='number of std deviations above the mean used for thresholding attention map')

args = parser.parse_args()

zoom_enhance(
    args.model_path,
    args.csv_path,
    args.data_path,
    args.full_sz_path,
    args.output_path,
    num_devs=args.num_devs
)

# examples usage of this function
# zoom_enhance(
#     'weights/chip_weights/fold1_final.pt', 
#     'data/test.csv', 
#     'data/test_224.npy',
#     'data/fullsize/test',
#     'data/crop_data/fold1_test.npy'
# )