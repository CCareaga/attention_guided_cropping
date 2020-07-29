import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str,
                    help='path to csv file containing image metadata')

parser.add_argument('--data_path', type=str,
                    help='path to image files, there should be an entry in the csv for each image')

parser.add_argument('--output_path', type=str,
                    help='name of the output file to store the resulting numpy file')

args = parser.parse_args()

def center_crop_resize(csv_path, data_path, output_path, sz=224):
    csv = pd.read_csv(csv_path)
  
    resized_imgs = []
    for img_name in tqdm(csv['image_name']):
        img_path = f'{data_path}/{img_name}.jpg'
        img = Image.open(img_path)
      
        cropped = transforms.functional.center_crop(img, min(img.size))
      
        cropped.save(img_path)
      
        resized = transforms.functional.resize(cropped, (sz, sz))
        resized_imgs.append(np.array(resized))
  
    stacked = np.stack(resized_imgs, 0)
    np.save(output_path, stacked)

center_crop_resize(args.csv_path, args.data_path, args.output_path)
