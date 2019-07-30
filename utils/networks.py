from keras.models import Model
from keras.layers import Input, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam

def get(name, input_cols=176, input_rows=256, n_filters=16, loss_function='binary_crossentropy'):
  networks = dict(
    Unet=unet,
    ResUnet=resunet
  )

  return networks[name](input_cols, input_rows, n_filters, loss_function)

def unet(input_cols, input_rows, n_filters, loss_function):
  inputs = Input((input_rows, input_cols, 1))

  # roznice: kernel_initializer, dropout, batchnorm, kernelsize na transpose w oryginale 2x2
  # Contracting path
  conv1 = Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
  conv1 = Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  pool1 = Dropout(0.5)(pool1)

  conv2 = Conv2D(filters=n_filters*2, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
  conv2 = Conv2D(filters=n_filters*2, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  pool2 = Dropout(0.5)(pool2)

  conv3 = Conv2D(filters=n_filters*4, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
  conv3 = Conv2D(filters=n_filters*4, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  pool3 = Dropout(0.5)(pool3)

  conv4 = Conv2D(filters=n_filters*8, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
  conv4 = Conv2D(filters=n_filters*8, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
  pool4 = Dropout(0.5)(pool4)

  conv5 = Conv2D(filters=n_filters*16, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
  conv5 = Conv2D(filters=n_filters*16, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

  # expansive path
  up6 = Conv2DTranspose(filters=n_filters*8, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv5)
  up6 = concatenate([up6, conv4])
  up6 = Dropout(0.5)(up6)
  conv6 = Conv2D(filters=n_filters*8, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
  conv6 = Conv2D(filters=n_filters*8, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

  up7 = Conv2DTranspose(filters=n_filters*4, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv6)
  up7 = concatenate([up7, conv3])
  up7 = Dropout(0.5)(up7)
  conv7 = Conv2D(filters=n_filters*4, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
  conv7 = Conv2D(filters=n_filters*4, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

  up8 = Conv2DTranspose(filters=n_filters*2, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv7)
  up8 = concatenate([up8, conv2])
  up8 = Dropout(0.5)(up8)
  conv8 = Conv2D(filters=n_filters*2, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
  conv8 = Conv2D(filters=n_filters*2, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

  up9 = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv8)
  up9 = concatenate([up9, conv1])
  up9 = Dropout(0.5)(up9)
  conv9 = Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
  conv9 = Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

  outputs = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid') (conv9)

  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer=Adam(), loss=loss_function, metrics=['accuracy'])

  return model

def resunet():
  return 'resunet'