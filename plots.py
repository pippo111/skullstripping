import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

matplotlib.use("TkAgg")

def draw_aug_samples(images_train, masks_train, generator_args):
  seed = random.randint(0, 128)
  image_datagen = ImageDataGenerator(**generator_args)

  check_image_gen = image_datagen.flow(images_train, seed=seed, batch_size=1, shuffle=True)
  check_mask_gen = image_datagen.flow(masks_train, seed=seed, batch_size=1, shuffle=True)

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

def plot_sample(X, y, preds, binary_preds, ix=None):
  if ix is None:
    ix = random.randint(0, len(X) - 1)

  fig, ax = plt.subplots(1, 4, figsize=(20, 10))
  ax[0].imshow(X[ix, ..., 0], cmap='bone')
  ax[0].contour(y[ix].squeeze(), cmap='bone', levels=[0.5])
  ax[0].set_title('Scan image')

  ax[1].imshow(y[ix].squeeze(), cmap='bone',)
  ax[1].set_title('Scan mask')

  ax[2].imshow(preds[ix].squeeze(), cmap='bone', vmin=0, vmax=1)
  ax[2].contour(y[ix].squeeze(), colors='yellow', levels=[0.5])
  ax[2].set_title('Scan Predicted Mask')

  ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
  ax[3].set_title('Scan Predicted binary');

  plt.show()