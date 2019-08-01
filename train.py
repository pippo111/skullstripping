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
X_train, y_train = dataset.get_data(
  cfg.dataset['train_dir'],
  cfg.dataset['image_width'],
  cfg.dataset['image_height'],
  cfg.dataset['limit']
)
X_valid, y_valid = dataset.get_data(
  cfg.dataset['validation_dir'],
  cfg.dataset['image_width'],
  cfg.dataset['image_height'],
  cfg.dataset['limit'],
  validation=True
)

# Create train generator for data augmentation
image_datagen = ImageDataGenerator(**cfg.generator_args)
mask_datagen = ImageDataGenerator(**cfg.generator_args)

image_generator = image_datagen.flow(X_train, seed=cfg.model['seed'], batch_size=cfg.model['batch_size'], shuffle=True)
mask_generator = mask_datagen.flow(y_train, seed=cfg.model['seed'], batch_size=cfg.model['batch_size'], shuffle=True)
train_generator = zip(image_generator, mask_generator)

# Show random input data just for simple check
fig_title = 'Limit={}, Batch size={}, Loss: {}, Model name: {}'.format(
  cfg.dataset['limit'],
  cfg.model['batch_size'],
  cfg.model['loss_fn'],
  cfg.model['checkpoint']
)
plots.draw_aug_samples(X_train, y_train, cfg.generator_args, text=fig_title)

# Set the model
model = networks.get(
  name=cfg.model['arch'],
  loss_function=loss.get(cfg.model['loss_fn']),
  input_cols=cfg.dataset['image_width'],
  input_rows=cfg.dataset['image_height']
)
model.summary()

# Run the model
results = model.fit_generator(
  train_generator,
  steps_per_epoch=(len(X_train) // cfg.model['batch_size']),
  epochs=cfg.model['epochs'],
  callbacks=[
    TensorBoard(log_dir='logs/{}-{}'.format(time(), cfg.model['checkpoint'])),
    EarlyStopping(patience=6, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('models/{}.hdf5'.format(cfg.model['checkpoint']), verbose=1, save_best_only=True, save_weights_only=True)
  ],
  validation_data=(X_valid, y_valid)
)

# Plot output results
plots.draw_results_log(results, text=fig_title)