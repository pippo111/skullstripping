import argparse
import os

from keras.preprocessing.image import load_img

parser = argparse.ArgumentParser()

parser.add_argument('--trainset-dir', type=str, help='Directory with train image set', default='z_train')
parser.add_argument('--validationset-dir', type=str, help='Directory with validation image set', default='z_validate')
parser.add_argument('--limit', type=int, help='Limit trainset to first number of items')

args, extra = parser.parse_known_args()

trainset_img_dir = args.trainset_dir + '/img'
trainset_mask_dir = args.trainset_dir + '/mask'
validationset_dir = args.validationset_dir
limit = args.limit

for root, dirs, files in os.walk(trainset_img_dir):
  for file in files[:limit]:
    img = load_img(root + '/' + file, color_mode='grayscale')
    print(img)

