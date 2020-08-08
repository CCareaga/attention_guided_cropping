import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import argparse

def center_crop_resize(csv_path, data_path, output_path, sz=224):

    # read in the metadata csv file that provides the image paths
    csv = pd.read_csv(csv_path)
  
    # for each image path in the csv file
    resized_imgs = []
    for img_name in tqdm(csv['image_name']):

        # open the full size image using the provided path
        img_path = f'{data_path}/{img_name}.jpg'
        img = Image.open(img_path)
      
        # center crop the image using the short side length
        cropped = transforms.functional.center_crop(img, min(img.size))
        
        # write back the image to decrease the size of the high res images
        cropped.save(img_path)
      
        # resize the center cropped image to 224 x 224 and convert to array
        resized = transforms.functional.resize(cropped, (sz, sz))
        resized_imgs.append(np.array(resized))
  
    # write out the stack of image arrays
    stacked = np.stack(resized_imgs, 0)
    np.save(output_path, stacked)

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str,
                    help='path to csv file containing image metadata')

parser.add_argument('--data_path', type=str,
                    help='path to image files, there should be an entry in the csv for each image')

parser.add_argument('--output_path', type=str,
                    help='name of the output file to store the resulting numpy file')

args = parser.parse_args()

center_crop_resize(args.csv_path, args.data_path, args.output_path)

