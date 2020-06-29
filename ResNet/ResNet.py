# Implement plain and residual networks in TensorFlow.


import tensorflow as tf
import numpy as np


## Plain network

def conv_bn(X, num_filters, reg=1e-3, filter_size=3, stride=1, pad='same'):
  initializer = tf.initializers.VarianceScaling(scale=2.0)
  regularizer = tf.keras.regularizers.l2(reg)

  X = tf.keras.layers.Conv2D(num_filters, filter_size, stride, pad, 
                             kernel_regularizer=regularizer,
                             kernel_initializer=initializer)(X)
  X = tf.keras.layers.BatchNormalization()(X)
  return X


def conv_bn_relu(X, num_filters, reg=1e-3, filter_size=3, stride=1, pad='same'):
  initializer = tf.initializers.VarianceScaling(scale=2.0)
  regularizer = tf.keras.regularizers.l2(reg)

  X = tf.keras.layers.Conv2D(num_filters, filter_size, stride, pad, 
                             kernel_regularizer=regularizer,
                             kernel_initializer=initializer)(X)
  X = tf.keras.layers.BatchNormalization()(X)
  X = tf.keras.layers.Activation('relu')(X)
  return X


def PlainNet(n=3, filter_sizes=[16,32,64], reg=1e-4):
  """
  The plain architecture follows [He et al, Deep Residual Learning for Image Recognition]
  on CIFAR-10.

  Inputs:
  -n: the repeated number of each block
  -filter_sizes: a list of 3 filter sizes, each filter size corresponding to a block
  -reg: regularization strength

  Output:
  -model: TensorFlow keras model
  """
  X_input = tf.keras.Input(shape=(32,32,3))

  # conv1, output shape: (32,32,16)
  X = conv_bn_relu(X_input, num_filters=filter_sizes[0], reg=reg)

  # conv_block_1, output shape: (32,32,16)
  i = 0
  while i < n:
    X = conv_bn_relu(X, num_filters=filter_sizes[0], reg=reg)
    X = conv_bn_relu(X, num_filters=filter_sizes[0], reg=reg)
    i += 1

  # conv_block_2, output shape: (16,16,32)
  X = tf.keras.layers.ZeroPadding2D(padding=1)(X)
  X = conv_bn_relu(X, num_filters=filter_sizes[1], reg=reg, stride=2, pad='valid')
  X = conv_bn_relu(X, num_filters=filter_sizes[1], reg=reg)
  i = 0
  while i < n-1:
    X = conv_bn_relu(X, num_filters=filter_sizes[1], reg=reg)
    X = conv_bn_relu(X, num_filters=filter_sizes[1], reg=reg)
    i += 1

  # conv_block_3, output shape: (8,8,64)
  X = tf.keras.layers.ZeroPadding2D(padding=1)(X)
  X = conv_bn_relu(X, num_filters=filter_sizes[2], reg=reg, stride=2, pad='valid')
  X = conv_bn_relu(X, num_filters=filter_sizes[2], reg=reg)
  i = 0
  while i < n-1:
    X = conv_bn_relu(X, num_filters=filter_sizes[2], reg=reg)
    X = conv_bn_relu(X, num_filters=filter_sizes[2], reg=reg)
    i += 1

  # output shape: (64,1)
  X = tf.keras.layers.GlobalAveragePooling2D()(X)
  X = tf.keras.layers.Dense(10, 
                            kernel_regularizer=tf.keras.regularizers.l2(reg),
                            kernel_initializer=tf.initializers.VarianceScaling(scale=2.0))(X)

  model = tf.keras.Model(X_input, X)

  return model
  
  
## Residual network

def identity_block(X, num_filters, reg=1e-3):
  X_shortcut = X

  X = conv_bn_relu(X, num_filters, reg=reg)
  X = conv_bn(X, num_filters, reg=reg)

  X = tf.keras.layers.Add()([X, X_shortcut])
  X = tf.keras.layers.Activation('relu')(X)
  return X


def projection_block(X, num_filters, reg=1e-3):
  X_shortcut = X

  X = tf.keras.layers.ZeroPadding2D(padding=1)(X)
  X = conv_bn_relu(X, num_filters, stride=2, pad='valid', reg=reg)
  X = conv_bn(X, num_filters, reg=reg)

  X_shortcut = conv_bn(X_shortcut, num_filters, filter_size=1, stride=2, 
                       pad='valid', reg=reg)

  X = tf.keras.layers.Add()([X, X_shortcut])
  X = tf.keras.layers.Activation('relu')(X)
  return X


def ResNet(n=3, filter_sizes=[16,32,64], reg=0):
  """
  The residual architecture follows a mixed of projection and identity shortcuts 
  in [He et al, Deep Residual Learning for Image Recognition] on CIFAR-10.

  Inputs:
  -n: the repeated number of each block
  -filter_sizes: a list of 3 filter sizes, each filter size corresponding to a block
  -reg: regularization strength

  Output:
  -model: TensorFlow keras model
  """
  X_input = tf.keras.Input(shape=(32,32,3))

  # conv1, output shape: (32,32,16)
  X = conv_bn_relu(X_input, num_filters=filter_sizes[0], reg=reg)

  # res_block_1, output shape: (32,32,16)
  i = 0
  while i < n:
    X = identity_block(X, num_filters=filter_sizes[0], reg=reg)
    i += 1

  # res_block_2, output shape: (16,16,32)
  X = projection_block(X, num_filters=filter_sizes[1], reg=reg)
  i = 0
  while i < n-1:
    X = identity_block(X, num_filters=filter_sizes[1], reg=reg)
    i += 1

  # res_block_3, output shape: (8,8,64)
  X = projection_block(X, num_filters=filter_sizes[2], reg=reg)
  i = 0
  while i < n-1:
    X = identity_block(X, num_filters=filter_sizes[2], reg=reg)
    i += 1

  # output shape: (64,1)
  X = tf.keras.layers.GlobalAveragePooling2D()(X)
  X = tf.keras.layers.Dense(10, 
                            kernel_regularizer=tf.keras.regularizers.l2(reg),
                            kernel_initializer=tf.initializers.VarianceScaling(2.0))(X)

  model = tf.keras.Model(X_input, X)

  return model
