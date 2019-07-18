import argparse
import numpy as np

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
parser.add_argument('--validationset-dir', type=str, help='Directory with validation image set', default='z_validation')
parser.add_argument('--image-width', type=int, help='Image width', default=176)
parser.add_argument('--image-height', type=int, help='Image height', default=256)
parser.add_argument('--limit', type=int, help='Limit validation set to first number of items')
parser.add_argument('--model-name', type=str, help='File name for the model checkpoint to save', default='unet')
parser.add_argument('--loss-function', type=str, help='Loss function name', default='binary_crossentropy')
args, extra = parser.parse_known_args()

# Setting up basic parameters
validationset_dir = args.validationset_dir
image_width = args.image_width
image_height = args.image_height
limit = args.limit
model_name = args.model_name
loss = loss[args.loss_function]

fig_title = 'Limit={}, Loss: {}, Model name: {}'.format(limit, args.loss_function, model_name)

X_valid, y_valid = dataset.get_data(validationset_dir, image_width, image_height, limit)

y_valid = y_valid * 2

model = network.get_unet(image_height, image_width, loss)
# model = network.get_custom_unet(image_height, image_width, loss)
model.load_weights('models/weights.{}.hdf5'.format(model_name))

loss, acc = model.evaluate(X_valid, y_valid, verbose=1)
print('loss={}, acc={}'.format(loss, acc))

preds_val = model.predict(X_valid, verbose=1)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

combined_val = y_valid + preds_val_t

plots.plot_sample(X_valid, y_valid, preds_val, preds_val_t, combined_val, text=fig_title)
