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
