import argparse
import numpy as np

# Local imports
import config as cfg
from helpers import networks
from helpers import plots
from helpers import dataset
from helpers import loss

X_valid, y_valid = dataset.get_data(
  cfg.dataset['test_dir'],
  cfg.dataset['image_width'],
  cfg.dataset['image_height'],
  cfg.dataset['limit']
)
y_valid = y_valid * 2

model = networks.get(
  name=cfg.model['arch'],
  loss_function=loss.get(cfg.model['loss_fn']),
  input_cols=cfg.dataset['image_width'],
  input_rows=cfg.dataset['image_height']
)
model.load_weights('models/{}.hdf5'.format(cfg.model['checkpoint']))

loss, acc = model.evaluate(X_valid, y_valid, verbose=1)
print('loss={}, acc={}'.format(loss, acc))

preds_val = model.predict(X_valid, verbose=1)
preds_val_t = (preds_val > cfg.model['threshold']).astype(np.uint8)

combined_val = y_valid + preds_val_t

yellow_perc = 0
reds_total = 0
for image in combined_val:
  r = np.count_nonzero(image == 1.0) # false negative (red)
  y = np.count_nonzero(image == 2.0) # false positive (yellow)
  g = np.count_nonzero(image == 3.0) # true positive (green)
  mask = g + y
  
  yellow_perc += 0 if mask == 0 else y/mask
  reds_total += r

print('False positive avg % = ', '{:.0%}'.format(yellow_perc / combined_val.shape[0]))
print('False negative total = ', reds_total)

fig_title = 'Limit={}, Loss fn: {}, Threshold: {}, Model name: {}, Loss={}, Acc={}'.format(
  cfg.dataset['limit'] or len(X_valid),
  cfg.model['loss_fn'],
  cfg.model['threshold'],
  cfg.model['checkpoint'],
  loss,
  acc
)

for slice_no in cfg.output['slice_numbers']:
  plots.plot_sample(
    X_valid,
    y_valid,
    preds_val,
    preds_val_t,
    combined_val,
    text=fig_title,
    ix=slice_no,
    threshold=cfg.model['threshold'],
    save_slice=cfg.output['save_slice'],
    slice_name=cfg.model['checkpoint']
  )
