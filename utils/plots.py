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

def plot_sample(X, y, preds, binary_preds, combined_preds, ix=None):
  if ix is None:
    ix = random.randint(0, len(X) - 1)

  fig, ax = plt.subplots(2, 3, figsize=(20, 10))
  ax[0][0].imshow(X[ix, ..., 0], cmap='gray')
  ax[0][0].set_title('Original image')

  ax[1][0].imshow(y[ix].squeeze(), cmap='gray',)
  ax[1][0].set_title('Original mask')

  ax[0][1].imshow(preds[ix].squeeze(), cmap='gray', vmin=0, vmax=1)
  ax[0][1].contour(preds[ix].squeeze(), colors='yellow', levels=[0.5])
  ax[0][1].set_title('Predicted Mask')

  ax[1][1].imshow(binary_preds[ix].squeeze(), cmap='gray')
  ax[1][1].set_title('Predicted Mask binary')

  cmap = matplotlib.colors.ListedColormap(['black', 'red', 'yellow', 'green'])
  norm = matplotlib.colors.Normalize(vmin=0, vmax=3)
  ax[0][2].imshow(combined_preds[ix].squeeze(), cmap=cmap, norm=norm)
  ax[0][2].set_title('Combined masks')

  ax[1][2].imshow(X[ix, ..., 0], cmap='gray')
  ax[1][2].imshow(combined_preds[ix].squeeze(), cmap=cmap, norm=norm, alpha=0.2)
  ax[1][2].set_title('Overlay mask')

  plt.show()