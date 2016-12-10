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
from moviepy.editor import ImageSequenceClip
import cv2

# TRAIN_DATA_FOLDER = '/Users/pavlobashmakov/code/sdc/behavioral-cloning/train1-complete'
TRAIN_DATA_FOLDER = '/Users/pavlobashmakov/code/sdc/behavioral-cloning/train2-complete'

  # print("data_folder =", data_folder)

def read_data_gen(data, labels, batch_size=64):
  size = len(data)
  begin = 0
  # for begin in range(0, size, batch_size):
  while True:
    end = begin + batch_size
    if end > size:
        begin = 0
        end = batch_size
    # print("gen = %d : %d" % (begin, end))
    yield data[begin:end], labels[begin:end]
    begin += batch_size

def pump_image_data(data):
  data_img = [np.asarray(Image.open(img_file)) for img_file in data]
  data_img = normalize(np.asarray(data_img))
  return data_img

def to_image_gen(data_gen):
  for X_batch_files, y_batch in data_gen:
    # X_image = [np.asarray(Image.open(img_file)) for img_file in X_batch]
    X_image = pump_image_data(X_batch_files)
    # y_image = y_batch
    # X_image = np.asarray(X_image)
    y_image = np.asarray(y_batch)
    # print('im =', X_image[0].shape)
    # print('im2 =', X_image[0])
    # for img_file in X_batch:
    #   im = Image.open(img_file)
    #   im_as_array = np.asarray(im)
    #   print('im =', im_as_array.shape)
    # print('%d : %d' % (len(X_batch), len(y_batch)))
    # print(X_batch[:3])
    # print(y_batch[:3])
    yield X_image, y_image

h, w, ch = 160, 320, 3
cfg_batch_size = 20
cfg_max_epoch = 2


def create_model_linear():
  a = Input(shape=(h, w, ch))
  f = Flatten()(a)

  # Fully Connected
  f = Dense(128)(f)
  f = Activation('tanh')(f)

  b = Dense(1)(f)
  model = Model(input=a, output=b)
  return model

def create_model_conv():
    nb_filters1 = 32
    nb_filters2 = 64
    kernel_size = (3, 3)
    pool_size = (2, 2)

    a = Input(shape=(h, w, ch))

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
    f = Dense(128)(f)
    f = Activation('tanh')(f)

    # f = Dropout(0.5)(f)

    b = Dense(1)(f)
    # b = Activation('sigmoid')(f)
    model = Model(input=a, output=b)
    return model


# Created model for linear regression
def create_model():
  model = create_model_conv()
  return model


X_data_files, y_data = bc_read_data(TRAIN_DATA_FOLDER)

print('len X_data_files =', len(X_data_files))
print('len y_data =', len(y_data))


'''
print('Remove jerky sections ...')

# Idxs to remove from dataset (bad driver:))
to_remove = [
  [0, 30],
  [300, 320],
  [900, 965],
  [1546, 1575],
  [2020, 2040],
  [2170, 2200],
  [3470, 3538]
]

def leave_elements_idx(n, to_remove):
  if len(to_remove) == 0: return np.arange(n)
  all_list = []
  for rm in to_remove:
      rm_arr = np.arange(rm[0], rm[1])
      all_list.append(rm_arr)
  conc = np.concatenate(all_list, axis = 0)
  return np.delete(np.arange(n), conc)

rm_idx = leave_elements_idx(len(X_data_files), to_remove)

X_data_files = np.asarray(X_data_files)
X_data_files = X_data_files[rm_idx]
X_data_files = X_data_files.tolist()
y_data = y_data[rm_idx]

print('len X_data_files =', len(X_data_files))
print('len y_data =', len(y_data))
'''



print('Split Train/Val/Tetst')
X_train_files, X_val_files, y_train, y_val = train_test_split(X_data_files, y_data, test_size=0.2, random_state=13)
X_val_files, X_test_files, y_val, y_test = train_test_split(X_val_files, y_val, test_size=0.5, random_state=17)

# X_train_files = np.asarray(X_train)
# X_val_files = np.asarray(X_val)
# X_test_files = np.asarray(X_test)

X_val = pump_image_data(X_val_files)
X_test = pump_image_data(X_test_files)

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
y_test = np.asarray(y_test)

# print('len X_train =', len(X_train))
# print('len y_train =', len(y_train))
# print('len X_val =', len(X_val))
# print('len y_val =', len(y_val))
# print('len X_test =', len(X_test))
# print('len y_test =', len(y_test))

print('X_train =', len(X_train_files))
print('y_train =', len(y_train))
print('X_val =', X_val.shape)
print('y_val =', y_val.shape)
print('X_test =', X_test.shape)
print('y_test =', y_test.shape)

# print(X_val[[1], :, :, :].shape)

# Prepare data generateors
data_gen = read_data_gen(X_train_files, y_train, batch_size = cfg_batch_size)
image_gen = to_image_gen(data_gen)

# Test Generator
# i = 0
# for X_batch, y_batch in image_gen:
#   print('i =', i)
#   print('X_batch.shape =', X_batch.shape)
#   print('y_batch.shape =', y_batch.shape)
#   i += 1
#   if i > 10: break



def restore_model(model_file):
  with open(model_file, 'r') as jfile:
    model = model_from_json(jfile.read())
  adam = Adam(decay=1e-3)
  model.compile(optimizer=adam, loss="mse")
  weights_file = model_file.replace('json', 'h5')
  model.load_weights(weights_file)
  return model

def save_model(model, model_file_name):
  # Save Model
  json_string = model.to_json()
  with open(model_file_name, 'w') as model_file:
    model_file.write(json_string)
  weights_file = model_file_name.replace('json', 'h5')
  # Save Weights
  model.save_weights(weights_file)


def create_train_model(train_gen, validation_data = (X_val, y_val), samples_per_epoch = None):
  model = create_model()
  adam = Adam(decay=1e-3)

  model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

  # X_train1 = pump_image_data(X_train_files)
  # y_train1 = y_train
  #
  # X_train1 = X_train1[:400]
  # y_train1 = y_train1[:400]
  # history = model.fit(X_train1, y_train1, batch_size=20, nb_epoch = cfg_max_epoch, verbose = 1, validation_split = 0.15)


  # history = model.fit(X_train, y_train, batch_size=20, nb_epoch = cfg_max_epoch, verbose = 1, validation_data = (X_val, y_val))

  samples_per_epoch = (samples_per_epoch // cfg_batch_size) * cfg_batch_size

  history = model.fit_generator(train_gen, samples_per_epoch = samples_per_epoch, nb_epoch = cfg_max_epoch, verbose = 1, validation_data = (X_val, y_val))

  print('history =', history.history)
  print("metrics_name =", model.metrics_names)


  return model

# ===============================================
# =============== Main Program ==================
# ===============================================

use_trained_model = False

if use_trained_model:
  # model = restore_model('models/model-1.json')
  model = restore_model('model.json')
else:
  model = create_train_model(image_gen, validation_data = (X_val, y_val), samples_per_epoch = len(X_train_files))
  # model.compile(loss='mean_squared_error', optimizer='adam')


# print("Evaluate model on test data: ")
# score = model.evaluate(X_test, y_test, verbose=1)
# print('Test score:', score)

if not use_trained_model:
  save_model(model, 'model.json')

# sample = X_val[:20, :, :, :]
# labels_sample = y_val[:20]


sample = pump_image_data(X_train_files[:40])
labels_sample = y_train[:40]
steering_angle = model.predict(sample, batch_size=1)
print('predicted steering_angle =', steering_angle)
print('labels =', labels_sample)



# print("X_data_files =", X_data_files[:10])


# print('X_data_files =', X_data_files)

'''
cfg_fps = 10
def process_image(get_frame, t):
  radius = 30
  idx = int(round(t * cfg_fps))
  if idx >= len(y_data):
    return get_frame(t)
  angle = y_data[idx]
  # print('t =', t, 'idx =', idx, ' y_data =', y_data[idx])
  image = get_frame(t)
  img = np.copy(image)
  cv2.rectangle(img, (w // 2 - w // 4, 0), (w // 2 + w // 4, h), [0, 0, 255], thickness = 2 )
  cv2.circle(img, (w // 2, h), radius, [255, 255, 255], thickness = 10 )
  cv2.circle(img, (w // 2 + round(radius * math.sin(angle)), h - round(radius * math.cos(angle))), 3, [255, 0, 0], thickness = 3 )
  cv2.putText(img, "{}".format(idx), (w // 2 + 80, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
  return img

clip = ImageSequenceClip(X_data_files, fps=cfg_fps)
clip = clip.fl(process_image)
clip.write_videofile('movie.mp4')
'''

# Save Model
# json_string = model.to_json()
# with open('model.json', 'w') as model_file:
#   model_file.write(json_string)
#
# # Save Weights
# model.save_weights('model.h5')

# print(X_train[:10])
# print(y_train[:10])

# Load and Exemine Data

# Define simplest Model and Train it on existing data

# Evaluate model performance

# Save model.json and model.h5 files from here.


if __name__ == '__main__':
  print("Hello there")
