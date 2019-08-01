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
