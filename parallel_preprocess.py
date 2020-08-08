import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import torchvision.transforms as transforms 
import multiprocessing

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
    # read in the metadata csv file that provides the image paths
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str,
                    help='path to csv file containing image metadata')

parser.add_argument('--data_path', type=str,
                    help='path to image files, there should be an entry in the csv for each image')

parser.add_argument('--output_path', type=str,
                    help='name of the output file to store the resulting numpy file')

parser.add_argument('--num_processes', type=int,
                    help='number of processes to be used for multiprocessing',
                    default=8)

parser.add_argument('--sz', type=int,
                    help='dimension of the processed image',
                    default=224)
args = parser.parse_args()

csv = pd.read_csv(args.csv_path)
data_path = args.data_path
output_path = args.output_path
sz = args.sz
num_processes = args.num_processes
  
    # for each image path in the csv file
resized_imgs = []
#for img_name in tqdm(csv['image_name']):
def process(img_name):
    print(img_name)
    sys.stdout.flush()
    # open the full size image using the provided path
    img_path = f'{data_path}/{img_name}.jpg'
    img = Image.open(img_path)
  
    # center crop the image using the short side length
    cropped = transforms.functional.center_crop(img, min(img.size))
    
    # write back the image to decrease the size of the high res images
    cropped.save(img_path)
  
    # resize the center cropped image to 224 x 224 and convert to array
    resized = transforms.functional.resize(cropped, (sz, sz))
    return np.array(resized)

with multiprocessing.Pool(num_processes) as p:
   resized_imgs = p.map(process, csv['image_name'].values.tolist())
# write out the stack of image arrays
stacked = np.stack(resized_imgs, 0)
np.save(output_path, stacked)


