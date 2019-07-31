from keras.models import Model
from keras.layers import Input, Dropout, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, Add
from keras.optimizers import Adam

def get(name, input_cols=176, input_rows=256, n_filters=16, loss_function='binary_crossentropy'):
  networks = dict(
    Unet=unet,
    ResUnet=resunet
  )

  return networks[name](input_cols, input_rows, n_filters, loss_function)

def unet(input_cols, input_rows, n_filters, loss_function):

  def conv_block(inputs, filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal'):
    x = Conv2D(
      filters=n_filters,
      kernel_size=kernel_size,
      activation=activation,
      padding=padding,
      kernel_initializer=kernel_initializer
    )(inputs)

    x = Conv2D(
      filters=n_filters,
      kernel_size=kernel_size,
      activation=activation,
      padding=padding,
      kernel_initializer=kernel_initializer
    )(x)

    return x

  inputs = Input((input_rows, input_cols, 1))

  # Contracting path
  conv1 = conv_block(inputs, n_filters)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = conv_block(pool1, n_filters*2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = conv_block(pool2, n_filters*4)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = conv_block(pool3, n_filters*8)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  # Bridge
  conv5 = conv_block(pool4, n_filters*16)

  # Expansive path
  up6 = Conv2DTranspose(filters=n_filters*8, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv5)
  print(conv1.shape, conv2.shape, conv3.shape, conv4.shape, conv5.shape, up6.shape)
  up6 = concatenate([up6, conv4])
  conv6 = conv_block(up6, n_filters*8)

  up7 = Conv2DTranspose(filters=n_filters*4, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv6)
  up7 = concatenate([up7, conv3])
  conv7 = conv_block(up7, n_filters*4)

  up8 = Conv2DTranspose(filters=n_filters*2, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv7)
  up8 = concatenate([up8, conv2])
  conv8 = conv_block(up8, n_filters*2)

  up9 = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv8)
  up9 = concatenate([up9, conv1])
  conv9 = conv_block(up9, n_filters)

  outputs = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv9)

  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer=Adam(), loss=loss_function, metrics=['accuracy'])

  return model

def resunet(input_cols, input_rows, n_filters, loss_function):
  # Convolutional block: BN -> ReLU -> Conv
  def conv_block(
    inputs,
    filters=n_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation='relu',
    batch_norm=True,
    padding='same',
    kernel_initializer='he_normal'
  ):
    if batch_norm:
      outputs = BatchNormalization()(inputs)
    else:
      outputs = inputs

    if activation:
      outputs = Activation('relu')(outputs)

    outputs = Conv2D(
      filters=n_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      kernel_initializer='he_normal'
    )(outputs)

    return outputs

  # Residual Unit: Input -> ConvBlock -> ConvBlock -> Addition
  def res_unit(
    inputs,
    filters=n_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation='relu',
    batch_norm=True,
    padding='same',
    kernel_initializer='he_normal'
  ):
    short = inputs
    outputs = conv_block(inputs, n_filters, strides=strides, activation=None, batch_norm=False)
    outputs = conv_block(outputs, n_filters)
    short = conv_block(short, n_filters, strides=strides, activation=None)
    outputs = Add()([outputs, short])

    return outputs

  # Inputs
  inputs = Input((input_rows, input_cols, 1))

  # Encoding
  conv1 = res_unit(inputs, n_filters, activation=None, batch_norm=False)
  conv2 = res_unit(conv1, n_filters*2, strides=(2, 2))
  conv3 = res_unit(conv2, n_filters*4, strides=(2, 2))

  # Bridge
  conv4 = res_unit(conv3, n_filters*8, strides=(2, 2))

  # Decoding
  up5 = Conv2DTranspose(filters=n_filters*4, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv4)
  up5 = concatenate([up5, conv3])
  conv5 = res_unit(up5, n_filters*4)

  up6 = Conv2DTranspose(filters=n_filters*2, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv5)
  up6 = concatenate([up6, conv2])
  conv6 = res_unit(up6, n_filters*2)

  up7 = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv6)
  up7 = concatenate([up7, conv1])
  conv7 = res_unit(up7, n_filters)

  # Outputs
  outputs = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid') (conv7)

  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer=Adam(), loss=loss_function, metrics=['accuracy'])

  return model
