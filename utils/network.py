from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.optimizers import Adam

def get_unet(img_rows, img_cols, loss='binary_crossentropy'):
  inputs = Input((img_rows, img_cols, 1))
  conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
  conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

  up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
  conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
  conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

  up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
  conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
  conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

  up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
  conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
  conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

  up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
  conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
  conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

  outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
  model = Model(inputs=[inputs], outputs=[outputs])

  model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])
  return model

# Custom unet from tutorial
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
  # first layer
  x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
              padding='same')(input_tensor)
  if batchnorm:
      x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # second layer
  x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
              padding='same')(x)
  if batchnorm:
      x = BatchNormalization()(x)
  x = Activation('relu')(x)
  return x
  
def get_custom_unet(img_rows, img_cols, loss='binary_crossentropy', n_filters=16, dropout=0.5, batchnorm=True):
  input_img = Input((img_rows, img_cols, 1), name='img')
  # contracting path
  c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
  p1 = MaxPooling2D((2, 2)) (c1)
  p1 = Dropout(dropout*0.5)(p1)

  c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
  p2 = MaxPooling2D((2, 2)) (c2)
  p2 = Dropout(dropout)(p2)

  c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
  p3 = MaxPooling2D((2, 2)) (c3)
  p3 = Dropout(dropout)(p3)

  c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
  p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
  p4 = Dropout(dropout)(p4)
  
  c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
  
  # expansive path
  u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
  u6 = concatenate([u6, c4])
  u6 = Dropout(dropout)(u6)
  c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

  u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
  u7 = concatenate([u7, c3])
  u7 = Dropout(dropout)(u7)
  c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

  u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
  u8 = concatenate([u8, c2])
  u8 = Dropout(dropout)(u8)
  c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

  u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
  u9 = concatenate([u9, c1], axis=3)
  u9 = Dropout(dropout)(u9)
  c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
  
  outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
  model = Model(inputs=[input_img], outputs=[outputs])

  model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])
  return model