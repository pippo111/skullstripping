import argparse
import os
import numpy as np
from time import time

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Local imports
import network
import plots

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
plots.draw_aug_samples(X_train, y_train, generator_args)

# Set the model
model = network.get_unet(image_height, image_width)
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

plots.draw_results_log(results)