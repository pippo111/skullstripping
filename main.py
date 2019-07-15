import argparse
import os
import random
import numpy as np

import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img, img_to_array

parser = argparse.ArgumentParser()

parser.add_argument('--trainset-dir', type=str, help='Directory with train image set', default='z_train')
parser.add_argument('--validationset-dir', type=str, help='Directory with validation image set', default='z_validate')
parser.add_argument('--image-width', type=int, help='Image width', default=176)
parser.add_argument('--image-height', type=int, help='Image height', default=256)
parser.add_argument('--limit', type=int, help='Limit trainset to first number of items')

args, extra = parser.parse_known_args()

trainset_img_dir = args.trainset_dir + '/img'
validationset_dir = args.validationset_dir
image_width = args.image_width
image_height = args.image_height
limit = args.limit

def norm_img(img):
  x_img = img_to_array(img)
  return x_img / 255

def get_data(dir):
  files_gen = ((root_img, files) for root_img, dirs, files in os.walk(trainset_img_dir) if len(files))

  for root_img, files in files_gen:
    root_mask = root_img.replace('img', 'mask')
    images = np.zeros((limit or len(files), image_height, image_width, 1), dtype=np.float32)
    masks = np.zeros((limit or len(files), image_height, image_width, 1), dtype=np.float32)

    for i, file in enumerate(files[:limit]):
      img = load_img(root_img + '/' + file, color_mode='grayscale')
      images[i] = norm_img(img)

      mask = load_img(root_mask + '/' + file, color_mode='grayscale')
      masks[i] = norm_img(mask)

    return images, masks

def show_random_scan(images, masks):
  ax = plt.subplots(1, 2, figsize=(20, 10))

  random_img_idx = random.randint(0, len(images))

  ax[0].set_title('Scan')
  ax[0].imshow(images[random_img_idx, ..., 0], cmap='gray', interpolation='bilinear')
  ax[1].set_title('Mask')
  ax[1].imshow(masks[random_img_idx, ..., 0], cmap='gray', interpolation='bilinear')

  plt.show()

X, y = get_data(trainset_img_dir)
show_random_scan(X, y)
