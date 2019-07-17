import argparse
import os
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.optimizers import Adam

from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

def norm_img(img):
  x_img = img_to_array(img)
  return x_img / 255

def get_data(directory, limit):
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

def draw_random_samples(images_train, masks_train, datagen):
  seed = random.randint(0, 128)
  check_image_gen = datagen.flow(X_train, seed=seed, batch_size=1, shuffle=True)
  check_mask_gen = datagen.flow(y_train, seed=seed, batch_size=1, shuffle=True)

  fig, ax = plt.subplots(2, 6, figsize=(20, 10))
  for i in range(6):
    batch = check_image_gen.next()
    ax[0][i].set_title('Train image ex.')
    ax[0][i].imshow(batch[0, ..., 0], cmap='bone', interpolation='bilinear')
  for i in range(6):
    batch = check_mask_gen.next()
    ax[1][i].set_title('Mask image ex.')
    ax[1][i].imshow(batch[0, ..., 0], cmap='bone', interpolation='bilinear')

  plt.show()

def draw_results_log(results):
  plt.figure(figsize=(8, 8))
  plt.title("Learning curve")
  plt.plot(results.history["loss"], label="loss")
  plt.plot(results.history["val_loss"], label="val_loss")
  plt.plot(results.history["acc"], label="acc")
  plt.plot(results.history["val_acc"], label="val_acc")
  plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
  plt.xlabel("Epochs")
  plt.ylabel("log_loss")
  plt.legend()

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

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

matplotlib.use("TkAgg")

# Command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--trainset-dir', type=str, help='Directory with train image set', default='z_train')
parser.add_argument('--validationset-dir', type=str, help='Directory with validation image set', default='z_validation')
parser.add_argument('--image-width', type=int, help='Image width', default=176)
parser.add_argument('--image-height', type=int, help='Image height', default=256)
parser.add_argument('--batch-size', type=int, help='Batch size', default=32)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
parser.add_argument('--limit', type=int, help='Limit trainset to first number of items')
parser.add_argument('--no-augmentation', type=bool, help='Don\'t apply data augmentation', default=False)
parser.add_argument('--model-name', type=str, help='File name for the model checkpoint to save', default='unet')
args, extra = parser.parse_known_args()

# Setting up basic parameters
trainset_img_dir = args.trainset_dir + '/img'
validationset_img_dir = args.validationset_dir + '/img'
image_width = args.image_width
image_height = args.image_height
batch_size = args.batch_size
epochs = args.epochs
limit = args.limit
no_augmentation = args.no_augmentation
model_name = args.model_name
seed = 1

# Get image data from specified directory
X_train, y_train = get_data(trainset_img_dir, limit)
X_valid, y_valid = get_data(validationset_img_dir, limit // 4)

# Create train generator for data augmentation
generator_args = dict(
  horizontal_flip=True,
  vertical_flip=True,
  rotation_range=90,
  width_shift_range=20,
  height_shift_range=20,
  zoom_range=0.05
)
image_datagen = ImageDataGenerator(**generator_args)
mask_datagen = ImageDataGenerator(**generator_args)

image_generator = image_datagen.flow(X_train, seed=seed, batch_size=batch_size, shuffle=True)
mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=batch_size, shuffle=True)
train_generator = zip(image_generator, mask_generator)

# Show random input data just for simple check
draw_random_samples(X_train, y_train, image_datagen)

# Set the model
model = get_unet(image_height, image_width)
model.summary()
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

# Run the model
if no_augmentation:
  results = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_valid, y_valid),
    callbacks=[tensorboard]
  )
else:
  results = model.fit_generator(
    train_generator,
    steps_per_epoch=(len(X_train) // batch_size),
    epochs=epochs,
    callbacks=[
      tensorboard,
      EarlyStopping(patience=25, verbose=1),
      ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
      ModelCheckpoint('models/weights.{}.hdf5'.format(model_name), verbose=1, save_best_only=True, save_weights_only=True)
    ],
    validation_data=(X_valid, y_valid)
  )

draw_results_log(results)