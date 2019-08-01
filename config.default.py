#
# Dataset setup
#
# Limit validation set to first number of items
limit: None
# Directory with train image dataset
train_dir: 'z_train'
# Directory with validation image set
validation_dir: 'z_validation'
# Directory with test images
test_dir: 'z_test'
# Image width
image_width: 176
# Image height
image_height: 256

#
# Model setup
#
# Model architecture
arch: 'ResUnet'
# Number of epochs
epochs: 50
# Batch size
batch_size: 20
# Loss function name
loss_fn: 'binary_crossentropy'
# Seed for augmentation shuffle
seed: 1
# Threshold for binary output
threshold: 0.5

#
# Output result setup
#
# File name for the model checkpoint to save
model_name: 'axis1_resunet_binary'
# Select slice to show or save on output
slice_numbers: [20, 44]
# Save showed slice as png
save_slice: False

#
# Agrs for data augmentation
#
generator_args = dict(
  # horizontal_flip=True,
  # vertical_flip=True,
  rotation_range=5,
  width_shift_range=0.1,
  height_shift_range=0.1,
  # zoom_range=0.05
)