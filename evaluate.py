import argparse
import numpy as np

# Local imports
from utils import networks
from utils import plots
from utils import dataset
from utils import loss

# Command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--validationset-dir', type=str, help='Directory with validation image set', default='z_validation')
parser.add_argument('--image-width', type=int, help='Image width', default=176)
parser.add_argument('--image-height', type=int, help='Image height', default=256)
parser.add_argument('--limit', type=int, help='Limit validation set to first number of items')
parser.add_argument('--model-name', type=str, help='File name for the model checkpoint to save', default='unet')
parser.add_argument('--loss-function', type=str, help='Loss function name', default='binary_crossentropy')
parser.add_argument('--slice-numbers', type=str, help='Select slice to show or save on output')
parser.add_argument('--save-slice', type=bool, help='Save showed slice as png', default=False)
parser.add_argument('--threshold', type=float, help='Set threshold', default=0.5)
args, extra = parser.parse_known_args()

# Setting up basic parameters
validationset_dir = args.validationset_dir
image_width = args.image_width
image_height = args.image_height
limit = args.limit
model_name = args.model_name
slice_numbers = args.slice_numbers.split(',')
save_slice = args.save_slice
loss_fn = loss.get(args.loss_function)
threshold = args.threshold

X_valid, y_valid = dataset.get_data(validationset_dir, image_width, image_height, limit)
y_valid = y_valid * 2

network_name = 'ResUnet'
model = networks.get(name=network_name, loss_function=loss_fn)
model.load_weights('models/weights.{}.hdf5'.format(model_name))

loss, acc = model.evaluate(X_valid, y_valid, verbose=1)
print('loss={}, acc={}'.format(loss, acc))

preds_val = model.predict(X_valid, verbose=1)
preds_val_t = (preds_val > threshold).astype(np.uint8)

combined_val = y_valid + preds_val_t

fig_title = 'Limit={}, Loss fn: {}, Threshold: {}, Model name: {}, Loss={}, Acc={}'.format(limit or len(X_valid), args.loss_function, threshold, model_name, loss, acc)

for slice_no in slice_numbers:
  plots.plot_sample(
    X_valid,
    y_valid,
    preds_val,
    preds_val_t,
    combined_val,
    text=fig_title,
    ix=int(slice_no),
    threshold=threshold,
    save_slice=save_slice
  )
