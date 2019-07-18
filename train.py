import argparse

from time import time

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Local imports
from utils import network
from utils import plots
from utils import dataset
from utils import loss

loss = dict(
  binary_crossentropy='binary_crossentropy',
  dice_loss=loss.dice_coef_loss
)

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
parser.add_argument('--loss-function', type=str, help='Loss function name', default='binary_crossentropy')
parser.add_argument('--model-name', type=str, help='File name for the model checkpoint to save', default='unet')
args, extra = parser.parse_known_args()

# Setting up basic parameters
trainset_dir = args.trainset_dir
validationset_dir = args.validationset_dir
image_width = args.image_width
image_height = args.image_height
batch_size = args.batch_size
epochs = args.epochs
limit = args.limit
no_augmentation = args.no_augmentation
model_name = args.model_name
loss = loss[args.loss_function]
seed = 1

fig_title = 'Limit={}, Batch size={}, Loss: {}, Model name: {}'.format(limit, batch_size, args.loss_function, model_name)

# Get image data from specified directory
X_train, y_train = dataset.get_data(trainset_dir, image_width, image_height, limit)
X_valid, y_valid = dataset.get_data(validationset_dir, image_width, image_height, limit // 4)

# Create train generator for data augmentation
generator_args = dict(
  horizontal_flip=True,
  vertical_flip=True,
  rotation_range=5,
  width_shift_range=0.1,
  height_shift_range=0.1,
  zoom_range=0.05
)
image_datagen = ImageDataGenerator(**generator_args)
mask_datagen = ImageDataGenerator(**generator_args)

image_generator = image_datagen.flow(X_train, seed=seed, batch_size=batch_size, shuffle=True)
mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=batch_size, shuffle=True)
train_generator = zip(image_generator, mask_generator)

# Show random input data just for simple check
plots.draw_aug_samples(X_train, y_train, generator_args, text=fig_title)

# Set the model
model = network.get_unet(image_height, image_width, loss)
model.summary()
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

# Run the model
if no_augmentation:
  results = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[
      tensorboard,
      # EarlyStopping(patience=25, verbose=1),
      ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
      ModelCheckpoint('models/weights.{}.hdf5'.format(model_name), verbose=1, save_best_only=True, save_weights_only=True)
    ],
    validation_data=(X_valid, y_valid),
  )
else:
  results = model.fit_generator(
    train_generator,
    steps_per_epoch=(len(X_train) // batch_size),
    epochs=epochs,
    callbacks=[
      tensorboard,
      # EarlyStopping(patience=25, verbose=1),
      ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
      ModelCheckpoint('models/weights.{}.hdf5'.format(model_name), verbose=1, save_best_only=True, save_weights_only=True)
    ],
    validation_data=(X_valid, y_valid)
  )

plots.draw_results_log(results, text=fig_title)