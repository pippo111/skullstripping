import argparse
import numpy as np

# Local imports
import network
import dataset
import plots

# Command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--validationset-dir', type=str, help='Directory with validation image set', default='z_validation')
parser.add_argument('--image-width', type=int, help='Image width', default=176)
parser.add_argument('--image-height', type=int, help='Image height', default=256)
parser.add_argument('--limit', type=int, help='Limit validation set to first number of items')
parser.add_argument('--model-name', type=str, help='File name for the model checkpoint to save', default='unet')
args, extra = parser.parse_known_args()

# Setting up basic parameters
validationset_img_dir = args.validationset_dir + '/img'
image_width = args.image_width
image_height = args.image_height
limit = args.limit
model_name = args.model_name

X_valid, y_valid = dataset.get_data(validationset_img_dir, image_width, image_height, limit)

model = network.get_unet(image_height, image_width)
model.load_weights('models/weights.{}.hdf5'.format(model_name))

loss, acc = model.evaluate(X_valid, y_valid, verbose=1)
print('loss={}, acc={}'.format(loss, acc))

preds_val = model.predict(X_valid, verbose=1)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

plots.plot_sample(X_valid, y_valid, preds_val, preds_val_t, 5)
