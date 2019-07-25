import os
import numpy as np

from keras.preprocessing.image import load_img, img_to_array

def norm_img(img):
  x_img = img_to_array(img)
  return x_img / 255

def get_data(directory, image_width, image_height, limit, validation=False):
  img_dir = directory + '/img'
  mask_dir = directory + '/mask'

  if not os.path.exists(img_dir):
    print('Directory {} not found!'.format(img_dir))
    exit()

  if not os.path.exists(mask_dir):
    print('Directory {} not found!'.format(mask_dir))
    exit()

  files_gen = ((root_img, files) for root_img, dirs, files in os.walk(img_dir) if len(files))

  for root_img, files in files_gen:
    root_mask = root_img.replace('img', 'mask')

    limit = limit or len(files)
    matrix_size = limit // 4 if validation == True else limit
    images = np.zeros((matrix_size, image_height, image_width, 1), dtype=np.float32)
    masks = np.zeros((matrix_size, image_height, image_width, 1), dtype=np.float32)

    for i, file in enumerate(files[:matrix_size]):
      img = load_img(root_img + '/' + file, color_mode='grayscale')
      images[i] = norm_img(img)

      mask = load_img(root_mask + '/' + file, color_mode='grayscale')
      masks[i] = norm_img(mask)

    return images, masks