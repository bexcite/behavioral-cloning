# The script used to create and train the model

import numpy as np
import tensorflow as tf
import os
import csv
import json
import math
from sklearn.model_selection import train_test_split
from sdc_utils import bc_read_data, normalize
from PIL import Image
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
# from moviepy.editor import ImageSequenceClip
import cv2

h, w, ch = 160, 320, 3


def create_model_linear(resize_factor = 1.0):

  hh = int(h // resize_factor)
  ww = int(w // resize_factor)

  a = Input(shape=(h, w, ch))
  f = Flatten()(a)

  # Fully Connected
  f = Dense(128)(f)
  f = Activation('tanh')(f)

  b = Dense(1)(f)
  model = Model(input=a, output=b)
  return model



def create_model_conv(resize_factor = 1.0):
    nb_filters1 = 32
    nb_filters2 = 64
    kernel_size = (3, 3)
    pool_size = (2, 2)

    hh = int(h // resize_factor)
    ww = int(w // resize_factor)

    a = Input(shape=(hh, ww, ch))

    # Convolution 1
    f = Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
                          border_mode='valid',
                          input_shape=(h, w, ch))(a)
    f = Activation('tanh')(f)
    f = MaxPooling2D(pool_size=pool_size)(f)

    # Convolution 2
    f = Convolution2D(nb_filters2, kernel_size[0], kernel_size[1],
                          border_mode='valid',
                          input_shape=(h, w, ch))(f)
    f = Activation('tanh')(f)
    f = MaxPooling2D(pool_size=pool_size)(f)

    f = Dropout(0.5)(f)

    f = Flatten()(f)

    # Fully Connected 1
    f = Dense(128)(f)
    f = Activation('tanh')(f)

    # Fully Connected 2
    # f = Dense(128)(f)
    # f = Activation('tanh')(f)

    # f = Dropout(0.5)(f)

    b = Dense(1)(f)
    # b = Activation('sigmoid')(f)
    model = Model(input=a, output=b)
    return model

def create_model_conv2(resize_factor = 1.0):
    nb_filters1 = 32
    nb_filters2 = 64
    kernel_size = (5, 5)
    pool_size = (2, 2)

    hh = int(h // resize_factor)
    ww = int(w // resize_factor)

    a = Input(shape=(hh, ww, ch))

    # Convolution 1
    f = Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
                          border_mode='same',
                          input_shape=(h, w, ch))(a)
    f = Activation('tanh')(f)
    f = MaxPooling2D(pool_size=pool_size)(f)

    # Convolution 2
    f = Convolution2D(nb_filters2, kernel_size[0], kernel_size[1],
                          border_mode='same',
                          input_shape=(h, w, ch))(f)
    f = Activation('tanh')(f)
    f = MaxPooling2D(pool_size=pool_size)(f)

    f = Dropout(0.5)(f)

    f = Flatten()(f)

    # Fully Connected 1
    f = Dense(512)(f)
    f = Activation('tanh')(f)

    # Fully Connected 2
    # f = Dense(128)(f)
    # f = Activation('tanh')(f)

    # f = Dropout(0.5)(f)

    b = Dense(1)(f)
    # b = Activation('sigmoid')(f)
    model = Model(input=a, output=b)
    return model

def create_model_conv3(resize_factor = 1.0):
    # ala comma.ai model

    hh = int(h // resize_factor)
    ww = int(w // resize_factor)
    # print('hh = ', hh)
    # print('ww = ', ww)

    a = Input(shape=(hh, ww, ch))
    # print('a =', a)

    # Convolution 1
    f = Convolution2D(16, 8, 8,
                          border_mode='same',
                          subsample = (4, 4))(a)
    f = Activation('elu')(f)

    # Convolution 2
    f = Convolution2D(32, 5, 5,
                          border_mode='same',
                          subsample = (2, 2))(f)
    f = Activation('elu')(f)

    # Convolution 3
    f = Convolution2D(64, 5, 5,
                          border_mode='same',
                          subsample = (2, 2))(f)

    f = Flatten()(f)
    f = Dropout(0.5)(f)
    f = Activation('elu')(f)

    # Fully Connected 1
    f = Dense(512)(f)
    f = Dropout(0.5)(f)
    f = Activation('elu')(f)

    # Fully Connected 2
    # f = Dense(128)(f)
    # f = Activation('tanh')(f)

    # f = Dropout(0.5)(f)

    b = Dense(1)(f)
    # b = Activation('sigmoid')(f)
    model = Model(input=a, output=b)
    return model

def create_model_conv4(resize_factor = 1.0):
    # ala comma.ai model

    hh = int(h // resize_factor)
    ww = int(w // resize_factor)

    a = Input(shape=(hh, ww, ch))

    # Convolution 1
    f = Convolution2D(12, 4, 4,
                          border_mode='same',
                          subsample=(2, 2))(a)
    f = Activation('elu')(f)
    # f = MaxPooling2D(pool_size=(2, 2))(f)
    # f = Dropout(0.2)(f)

    # Convolution 2
    f = Convolution2D(24, 3, 3,
                          border_mode='same',
                          subsample=(2, 2))(f)
    f = Activation('elu')(f)
    # f = MaxPooling2D(pool_size=(2, 2))(f)
    # f = Dropout(0.2)(f)

    # Convolution 3
    f = Convolution2D(48, 3, 3,
                          border_mode='same',
                          subsample=(1, 1))(f)
    f = Activation('elu')(f)
    f = MaxPooling2D(pool_size=(2, 2))(f)

    # Convolution 4
    f = Convolution2D(96, 3, 3,
                          border_mode='same',
                          subsample=(1, 1))(f)
    f = Activation('elu')(f)
    f = MaxPooling2D(pool_size=(2, 2))(f)


    # f = Dropout(0.5)(f)

    f = Flatten()(f)

    # Fully Connected 1
    f = Dense(512)(f)
    # f = Dropout(0.5)(f)
    f = Activation('elu')(f)

    # Fully Connected 2
    f = Dense(64)(f)
    # f = Dropout(0.5)(f)
    f = Activation('elu')(f)

    # Fully Connected 2
    # f = Dense(128)(f)
    # f = Activation('tanh')(f)

    # f = Dropout(0.5)(f)

    b = Dense(1)(f)
    # b = Activation('sigmoid')(f)
    model = Model(input=a, output=b)
    return model

def create_model_conv5(resize_factor = 1.0, crop_bottom = None):
    # ala comma.ai model

    if crop_bottom:
      hh = h - crop_bottom
    else:
      hh = h

    hh = int(hh // resize_factor)
    ww = int(w // resize_factor)
    print('model hh = ', hh)
    print('model ww = ', ww)

    dropout = 0.5 / resize_factor
    print('model dropout = ', dropout)

    a = Input(shape=(hh, ww, ch))

    # f = Convolution2D(3, 1, 1,
    #                   border_mode='valid',
    #                   subsample=(1, 1))(a)

    # Convolution 1
    f = Convolution2D(32, 4, 4,
                          border_mode='valid',
                          subsample=(2, 2))(a)
    f = Activation('elu')(f)
    # f = MaxPooling2D(pool_size=(2, 2))(f)
    f = Dropout(dropout)(f)

    # Convolution 2
    f = Convolution2D(64, 3, 3,
                          border_mode='valid',
                          subsample=(2, 2))(f)
    f = Activation('elu')(f)
    # f = MaxPooling2D(pool_size=(2, 2))(f)
    f = Dropout(dropout)(f)

    # Convolution 3
    f = Convolution2D(128, 3, 3,
                          border_mode='valid',
                          subsample=(1, 1))(f)
    f = Activation('elu')(f)
    f = MaxPooling2D(pool_size=(2, 2))(f)

    # Convolution 4
    # f = Convolution2D(96, 3, 3,
    #                       border_mode='valid',
    #                       subsample=(1, 1))(f)
    # f = Activation('elu')(f)
    # f = MaxPooling2D(pool_size=(2, 2))(f)



    f = Dropout(dropout)(f)

    f = Flatten()(f)

    # Fully Connected 1
    f = Dense(512)(f)
    f = Dropout(dropout)(f)
    f = Activation('elu')(f)

    # Fully Connected 2
    f = Dense(64)(f)
    # f = Dropout(dropout)(f)
    f = Activation('elu')(f)

    # Fully Connected 2
    # f = Dense(128)(f)
    # f = Activation('tanh')(f)

    # f = Dropout(0.5)(f)

    b = Dense(1)(f)
    # b = Activation('sigmoid')(f)
    model = Model(input=a, output=b)
    return model



# Created model for linear regression
def create_model(model_type = 'cnn', resize_factor = 1.0, crop_bottom = None):
  models = {
    'linear' : create_model_linear,
    'cnn': create_model_conv,
    'cnn2': create_model_conv2,
    'cnn3': create_model_conv3,
    'cnn4': create_model_conv4,
    'cnn5': create_model_conv5
  }
  builder = models.get(model_type)
  model = builder(resize_factor, crop_bottom)
  return model



def restore_model(model_file):
  with open(model_file, 'r') as jfile:
    model = model_from_json(jfile.read())
  adam = Adam(decay=1e-3)
  model.compile(optimizer=adam, loss="mse")
  weights_file = model_file.replace('json', 'h5')
  model.load_weights(weights_file)
  return model
