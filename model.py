# The script used to create and train the model

import numpy as np
import tensorflow as tf
import os
import csv
from sklearn.model_selection import train_test_split
from sdc_utils import bc_read_data
from PIL import Image

TRAIN_DATA_FOLDER = '/Users/pavlobashmakov/code/sdc/behavioral-cloning/train1-complete'

  # print("data_folder =", data_folder)

def read_data_gen(data, labels, batch_size=64):
  size = len(data)
  for begin in range(0, size, batch_size):
    end = begin + batch_size
    if end < size:
      yield data[begin:end], labels[begin:end]

def to_image_gen(data_gen):
  for X_batch, y_batch in data_gen:
    X_image = [np.asarray(Image.open(img_file)) for img_file in X_batch]
    X_image = np.asarray(X_image)
    y_image = np.asarray(y_batch)
    # print('im =', X_image[0].shape)
    # print('im2 =', X_image.shape)
    # for img_file in X_batch:
    #   im = Image.open(img_file)
    #   im_as_array = np.asarray(im)
    #   print('im =', im_as_array.shape)
    # print('%d : %d' % (len(X_batch), len(y_batch)))
    # print(X_batch[:3])
    # print(y_batch[:3])
    yield X_image, y_image


X_data, y_data = bc_read_data(TRAIN_DATA_FOLDER)

print('len X_data =', len(X_data))
print('len y_data =', len(y_data))

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.4, random_state=13)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=17)

print('len X_train =', len(X_train))
print('len y_train =', len(y_train))
print('len X_val =', len(X_val))
print('len y_val =', len(y_val))
print('len X_test =', len(X_test))
print('len y_test =', len(y_test))

data_gen = read_data_gen(X_train, y_train, batch_size = 200)
image_gen = to_image_gen(data_gen)

i = 0
for X_batch, y_batch in image_gen:
  print('i =', i)
  print('X_batch.shape =', X_batch.shape)
  print('y_batch.shape =', y_batch.shape)
  i += 1

# print(X_train[:10])
# print(y_train[:10])

# Load and Exemine Data

# Define simplest Model and Train it on existing data

# Evaluate model performance

# Save model.json and model.h5 files from here.


if __name__ == '__main__':
  print("Hello there")
