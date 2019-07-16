import argparse
import os
import random
import numpy as np

import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.optimizers import Adam

from time import time
from keras.callbacks import TensorBoard

def norm_img(img):
  x_img = img_to_array(img)
  return x_img / 255

def get_data(directory):
  files_gen = ((root_img, files) for root_img, dirs, files in os.walk(directory) if len(files))

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

def show_random_scan(images_train, masks_train, images_valid, masks_valid):
  fig, ax = plt.subplots(2, 2, figsize=(20, 10))

  random_img_idx = random.randint(0, len(images_train))

  ax[0][0].set_title('Scan from train set')
  ax[0][0].imshow(images_train[random_img_idx, ..., 0], cmap='gray', interpolation='bilinear')
  ax[0][1].set_title('Mask from train set')
  ax[0][1].imshow(masks_train[random_img_idx, ..., 0], cmap='gray', interpolation='bilinear')

  ax[1][0].set_title('Scan from validation set')
  ax[1][0].imshow(images_valid[random_img_idx, ..., 0], cmap='gray', interpolation='bilinear')
  ax[1][1].set_title('Mask from validation set')
  ax[1][1].imshow(masks_valid[random_img_idx, ..., 0], cmap='gray', interpolation='bilinear')

  plt.show()

def get_unet(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

    # conv10 = Conv2D(2, (1, 1), activation='softmax')(conv9)
    # conv10 = Conv2D(4, (1, 1), activation='sigmoid')(conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)
    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

parser = argparse.ArgumentParser()

parser.add_argument('--trainset-dir', type=str, help='Directory with train image set', default='z_train')
parser.add_argument('--validationset-dir', type=str, help='Directory with validation image set', default='z_validation')
parser.add_argument('--image-width', type=int, help='Image width', default=176)
parser.add_argument('--image-height', type=int, help='Image height', default=256)
parser.add_argument('--limit', type=int, help='Limit trainset to first number of items')

args, extra = parser.parse_known_args()

trainset_img_dir = args.trainset_dir + '/img'
validationset_img_dir = args.validationset_dir + '/img'
image_width = args.image_width
image_height = args.image_height
limit = args.limit

X_train, y_train = get_data(trainset_img_dir)
X_valid, y_valid = get_data(validationset_img_dir)

show_random_scan(X_train, y_train, X_valid, y_valid)

model = get_unet(image_height, image_width)
model.summary()

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

results = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tensorboard])