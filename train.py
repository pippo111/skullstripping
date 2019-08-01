from time import time
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Local imports
import config as cfg
from helpers import networks
from helpers import plots
from helpers import dataset
from helpers import loss

# Get image data from specified directory
X_train, y_train = dataset.get_data(cfg.train_dir, cfg.image_width, cfg.image_height, cfg.limit)
X_valid, y_valid = dataset.get_data(cfg.validation_dir, cfg.image_width, cfg.image_height, cfg.limit, validation=True)

# Create train generator for data augmentation
image_datagen = ImageDataGenerator(**cfg.generator_args)
mask_datagen = ImageDataGenerator(**cfg.generator_args)

image_generator = image_datagen.flow(X_train, seed=cfg.seed, batch_size=cfg.batch_size, shuffle=True)
mask_generator = mask_datagen.flow(y_train, seed=cfg.seed, batch_size=cfg.batch_size, shuffle=True)
train_generator = zip(image_generator, mask_generator)

# Show random input data just for simple check
fig_title = 'Limit={}, Batch size={}, Loss: {}, Model name: {}'.format(cfg.limit, cfg.batch_size, cfg.loss_fn, cfg.model_name)
plots.draw_aug_samples(X_train, y_train, cfg.generator_args, text=fig_title)

# Set the model
model = networks.get(name=cfg.arch, loss_function=cfg.loss_fn)
model.summary()

# Run the model
results = model.fit_generator(
  train_generator,
  steps_per_epoch=(len(X_train) // cfg.batch_size),
  epochs=cfg.epochs,
  callbacks=[
    TensorBoard(log_dir='logs/{}-{}'.format(time(), cfg.model_name)),
    EarlyStopping(patience=6, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('models/weights.{}.hdf5'.format(cfg.model_name), verbose=1, save_best_only=True, save_weights_only=True)
  ],
  validation_data=(X_valid, y_valid)
)

# Plot output results
plots.draw_results_log(results, text=fig_title)