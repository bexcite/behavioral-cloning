'''
Trains a given model on a given dataset.

'''

import cv2
import argparse
import numpy as np
import time
import os
# import math
# from moviepy.editor import ImageSequenceClip
from sklearn.model_selection import train_test_split
from sdc_utils import bc_read_data, normalize, pump_image_data
from sdc_utils import read_data_gen, read_image_gen
from sdc_utils import load_dataset, load_all_datasets
# from jerky_utils import remove_jerky_sections
from model import create_model, create_model_linear, create_model_conv
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEBUG = 0

def save_model(model, model_file_name):
  # Save Model
  json_string = model.to_json()
  with open(model_file_name, 'w') as model_file:
    model_file.write(json_string)
  weights_file = model_file_name.replace('json', 'h5')
  # Save Weights
  model.save_weights(weights_file)

def train_model_on_gen(model, train_gen,
      validation_data = None,
      samples_per_epoch = None,
      batch_size = None,
      nb_epoch = 1):

  adam = Adam(lr=1e-4, decay=0.1)
  model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

  # samples_per_epoch = (samples_per_epoch // batch_size) * batch_size

  save_time = time.strftime("%Y%m%d%H%M%S")

  model_saver = ModelCheckpoint(filepath="checkpoints/%s_weights_n%d_{epoch:02d}_{val_loss:.4f}.hdf5" % (save_time, samples_per_epoch), verbose=1, save_best_only=True)

  history = model.fit_generator(train_gen,
                                samples_per_epoch = samples_per_epoch,
                                nb_epoch = nb_epoch,
                                verbose = 1,
                                validation_data = validation_data,
                                callbacks=[model_saver])

  print('history =', history.history)
  print("metrics_name =", model.metrics_names)

  return model


def train_model(model, data, labels,
      validation_data = None,
      batch_size = None,
      nb_epoch = 1):

  adam = Adam(lr=1e-4, decay=0.1) # decay=0.3, lr=1e-3
  model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

  # samples_per_epoch = (samples_per_epoch // batch_size) * batch_size

  save_time = time.strftime("%Y%m%d%H%M%S")

  model_saver = ModelCheckpoint(filepath="checkpoints/%s_weights_n%d_{epoch:02d}_{val_loss:.4f}.hdf5" % (save_time, len(data)), verbose=1, save_best_only=True)
  history = model.fit(data, labels, nb_epoch = nb_epoch, verbose = 1, validation_data = validation_data, callbacks=[model_saver])

  print('history =', history.history)
  print("metrics_name =", model.metrics_names)

  return model


def main():

  # parse arguments
  parser = argparse.ArgumentParser(description="Trains model on provided dataset and model type")
  parser.add_argument('--dataset', type=str, help='dataset folder with csv file and image folders')
  parser.add_argument('--base_path', type=str, default='../../../sdc/behavioral-cloning', help='Base datasets path - required for "all" dataset')
  parser.add_argument('--model', type=str, default='linear', help='model to evaluate, current list: {linear, cnn}')
  parser.add_argument('--save_file', type=str, default='model.json', help='save model and params')
  parser.add_argument('--nb_epoch', type=int, default=1, help='# of training epoch')
  parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
  parser.add_argument('--restore_weights', type=str, help='Restore weights from checkpoint')
  parser.add_argument('--debug_mode', default=False, action='store_true', help='Turn on DEBUG mode')
  parser.add_argument('--validation_split', type=float, default=0.15, help='Validation split - used for val+test combined')

  args = parser.parse_args()

  dataset_path = args.dataset
  base_path = args.base_path
  save_file = args.save_file
  nb_epoch = args.nb_epoch
  batch_size = args.batch_size
  model_type = args.model
  restore_weights = args.restore_weights
  validation_split = args.validation_split

  # This is a bad style ... but ... heh ...
  global DEBUG

  if args.debug_mode:
    DEBUG = 1

  if DEBUG:
    print("DEBUG mode is ON")


  if not dataset_path:
    parser.error("Dataset is not specified")

  print('dataset = ', dataset_path)
  print('save_file = ', save_file)
  print('nb_epoch = ', nb_epoch)
  print('batch_size = ', batch_size)

  if dataset_path == 'all':
    print('Load ALL datasets.')
    X_data_files, y_data = load_all_datasets(base_path, remove_jerky = True)
  else:
    X_data_files, y_data = load_dataset(dataset_path, remove_jerky = True)

  print('len X_data_files =', len(X_data_files))
  print('len y_data =', len(y_data))

  '''
  y_arr = np.asarray(y_data)
  z = np.sum(y_arr == 0)
  print('z = ', z)
  neg = np.sum(y_arr < 0)
  print('neg = ', neg)
  pos = np.sum(y_arr > 0)
  print('pos = ', pos)

  '''


  print('Split Train/Val/Tetst')

  X_train_files, X_val_files, y_train, y_val = train_test_split(
      X_data_files, y_data,
      test_size=validation_split,
      random_state=13)

  X_val_files, X_test_files, y_val, y_test = train_test_split(
      X_val_files, y_val,
      test_size=0.5,
      random_state=17)

  X_val = pump_image_data(X_val_files)
  X_test = pump_image_data(X_test_files)

  y_train = np.asarray(y_train)
  y_val = np.asarray(y_val)
  y_test = np.asarray(y_test)

  print('X_train =', len(X_train_files))
  print('y_train =', len(y_train))
  print('X_val =', X_val.shape)
  print('y_val =', y_val.shape)
  print('X_test =', X_test.shape)
  print('y_test =', y_test.shape)


  if DEBUG:
    print("DEBUG: reducing data")
    # select_idxs = [0, 4, 10]
    # X_train_files = np.asarray(X_train_files)[select_idxs]
    # X_train_files = X_train_files.tolist()
    # y_train = y_train[select_idxs]
    X_train_files = X_train_files[:500]
    y_train = y_train[:500]
    print('X_train =', len(X_train_files))
    print('y_train =', len(y_train))
    X_val = X_val[:150]
    y_val = y_val[:150]
    print('X_val =', X_val.shape)
    print('y_val =', y_val.shape)

  # X_train_files = X_train_files[:100]
  # y_train = y_train[:100]
  # X_val = X_val[:40]
  # y_val = y_val[:40]

  # Prepare data generateors
  data_gen = read_data_gen(X_train_files, y_train, batch_size = batch_size)
  image_gen = read_image_gen(data_gen)

  print('Creating model.')
  model = create_model(model_type)

  if restore_weights:
    print('Restoring weights from ', restore_weights)
    model.load_weights(restore_weights)

  print('Train model ...')

  if not DEBUG:
    model = train_model_on_gen(model, image_gen,
                        validation_data = (X_val, y_val),
                        samples_per_epoch = len(X_train_files),
                        nb_epoch = nb_epoch,
                        batch_size = batch_size)
  else:
    # DEBUG
    X_train = pump_image_data(X_train_files)
    # print('X_train =', X_train)
    # print('y_train =', y_train)
    model = train_model(model, X_train, y_train,
                        validation_data = (X_val, y_val),
                        nb_epoch = nb_epoch,
                        batch_size = batch_size)

  print('Saving model to ', save_file)
  save_model(model, save_file)

  print('Inference on train data ...')

  sample = pump_image_data(X_train_files[:100] if len(X_train_files) > 100 else X_train_files)
  labels_sample = y_train[:100] if len(y_train) > 100 else y_train

  # if DEBUG:
  #   sample = pump_image_data(X_train_files)
  #   labels_sample = y_train
  steering_angle = model.predict(sample)
  print('predicted steering_angle =', steering_angle)
  print('labels =', labels_sample)
  rmse = np.sqrt(np.mean((steering_angle-labels_sample)**2))
  print("Train model evaluated RMSE:", rmse)

  save_time = time.strftime("%Y%m%d%H%M%S")

  def make_fig(model_type, rmse, pic_name, target_data, predict_data):
    plt.figure(figsize = (32, 8))
    plt.plot(target_data, 'r.-', label='target')
    plt.plot(predict_data, 'b^-', label='predict')
    plt.legend(loc='best')
    plt.title("Model type: %s, RMSE: %.2f" % (model_type, rmse))
    plt.savefig('graphs/%s' % pic_name)

  pic_name = "%s_%s_train.png" % (model_type, save_time)
  make_fig(model_type, rmse, pic_name, labels_sample, steering_angle)

  # Test only
  X_test = X_test[:100] if len(X_test) > 100 else X_test
  y_test = y_test[:100] if len(y_test) > 100 else y_test

  print('Evaluate model on test data batch.')
  test_predicts = model.predict(X_test)
  rmse = np.sqrt(np.mean((test_predicts-y_test)**2))
  print("Test model evaluated RMSE:", rmse)

  pic_name = "%s_%s_test.png" % (model_type, save_time)
  make_fig(model_type, rmse, pic_name, y_test, test_predicts)


if __name__ == '__main__':
  main()
