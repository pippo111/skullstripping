import argparse
import numpy as np

# Local imports
import config as cfg
from helpers import networks
from helpers import plots
from helpers import dataset
from helpers import loss

X_valid, y_valid = dataset.get_data(cfg.validation_dir, cfg.image_width, cfg.image_height, cfg.limit)
y_valid = y_valid * 2

model = networks.get(name=cfg.arch, loss_function=cfg.loss_fn)
model.load_weights('models/weights.{}.hdf5'.format(cfg.model_name))

loss, acc = model.evaluate(X_valid, y_valid, verbose=1)
print('loss={}, acc={}'.format(loss, acc))

preds_val = model.predict(X_valid, verbose=1)
preds_val_t = (preds_val > cfg.threshold).astype(np.uint8)

combined_val = y_valid + preds_val_t

fig_title = 'Limit={}, Loss fn: {}, Threshold: {}, Model name: {}, Loss={}, Acc={}'.format(cfg.limit or len(X_valid), cfg.loss_fn, cfg.threshold, cfg.model_name, loss, acc)

for slice_no in cfg.slice_numbers:
  plots.plot_sample(
    X_valid,
    y_valid,
    preds_val,
    preds_val_t,
    combined_val,
    text=fig_title,
    ix=int(slice_no),
    threshold=cfg.threshold,
    save_slice=cfg.save_slice
  )
