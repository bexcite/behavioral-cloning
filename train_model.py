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
from sdc_utils import extend_with_flipped
from sdc_utils import read_data_gen, read_image_gen
from sdc_utils import load_dataset, load_all_datasets, load_datasets
# from jerky_utils import remove_jerky_sections
from model import create_model, create_model_linear, create_model_conv
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from video import make_video

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
      lr = 0.001,
      samples_per_epoch = None,
      batch_size = None,
      nb_epoch = 1):

  adam = Adam(lr=lr, decay=0.1)
  model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

  # samples_per_epoch = (samples_per_epoch // batch_size) * batch_size

  save_time = time.strftime("%Y%m%d%H%M%S")

  model_saver = ModelCheckpoint(filepath="checkpoints/%s_weights_n%d_{epoch:02d}_{val_loss:.4f}.hdf5" % (save_time, samples_per_epoch), verbose=1, save_best_only=False)

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
      lr = 0.001,
      batch_size = None,
      nb_epoch = 1):

  adam = Adam(lr=lr, decay=0.1) # decay=0.3, lr=1e-3
  model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

  # samples_per_epoch = (samples_per_epoch // batch_size) * batch_size

  save_time = time.strftime("%Y%m%d%H%M%S")

  model_saver = ModelCheckpoint(filepath="checkpoints/%s_weights_n%d_{epoch:02d}_{val_loss:.4f}.hdf5" % (save_time, len(data)), verbose=1, save_best_only=False)
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
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate - default 0.001')
  parser.add_argument('--resize_factor', type=float, default=1, help='Resize image factor - default 1.0')
  parser.add_argument('--crop_bottom', type=int, default=0, help='Crop bottom. to remove car image')
  parser.add_argument('--remove_jerky', default=False, action='store_true', help='Remove jerky sections if dataset name is present in jerky_utils.py')
  parser.add_argument('--flip_images', default=False, action='store_true', help='Flip images of all training data. Increse the size of train by 2x')
  parser.add_argument('--left_right_images', default=False, action='store_true', help='Load left and right images into training data. Increse the size of train by 3x')

  args = parser.parse_args()

  dataset_path = args.dataset
  base_path = args.base_path
  save_file = args.save_file
  nb_epoch = args.nb_epoch
  batch_size = args.batch_size
  model_type = args.model
  restore_weights = args.restore_weights
  validation_split = args.validation_split
  remove_jerky = args.remove_jerky
  lr = args.lr
  resize_factor = args.resize_factor
  flip_images = args.flip_images
  flip_images_ratio = 0.3
  left_right_images = args.left_right_images
  crop_bottom = args.crop_bottom


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
  print('lr =', lr)
  print('resize_factor =', resize_factor)
  print('crop_bottom =', crop_bottom)
  print('flip_images =', flip_images)
  print('left_right_images =', left_right_images)

  if dataset_path == 'all':
    print('Load ALL datasets.')
    X_data_files, y_data = load_all_datasets(base_path, remove_jerky = remove_jerky, left_right = left_right_images)
  elif dataset_path == 'what':
    print('Load ... datasets.')
    ds = ['train1-complete', 'train2-complete', 'train4-complete', 'train6-complete', 'train9-complete', 'train10-complete', 'data']
    X_data_files, y_data = load_datasets(base_path, ds, remove_jerky = remove_jerky, left_right = left_right_images)
  elif dataset_path == 'corners':
    print('Load Corners datasets.')
    ds = ['data', 'corner2', 'corner3', 'train1-complete', 'train2-complete', 'train4-complete', 'train6-complete', 'train9-complete', 'train10-complete']
    X_data_files, y_data = load_datasets(base_path, ds, remove_jerky = remove_jerky, left_right = left_right_images)
  else:
    X_data_files, y_data = load_dataset(dataset_path, remove_jerky = remove_jerky, left_right = left_right_images)

  print('len X_data_files =', len(X_data_files))
  print('len y_data =', len(y_data))


  y_arr = np.asarray(y_data)
  z = np.sum(y_arr == 0)
  print('y == 0 = ', z)
  neg = np.sum(y_arr < 0)
  print('y < 0 = ', neg)
  pos = np.sum(y_arr > 0)
  print('y > 0 = ', pos)
  y_min = np.min(y_arr)
  print('y min = ', y_min)
  y_max = np.max(y_arr)
  print('y max = ', y_max)
  yn10 = np.sum(np.absolute(y_arr) < 0.1)
  print('y10 = ', yn10)
  yn50 = np.sum(np.absolute(y_arr) < 0.5)
  print('y50 = ', yn50)
  yn80 = np.sum(np.absolute(y_arr) > 0.8)
  print('y80 = ', yn80)


  print('Split Train/Val/Tetst')

  X_train_files, X_val_files, y_train, y_val = train_test_split(
      X_data_files, y_data,
      test_size=validation_split,
      random_state=13)

  X_val_files, X_test_files, y_val, y_test = train_test_split(
      X_val_files, y_val,
      test_size=0.5,
      random_state=17)

  X_val = pump_image_data(X_val_files, resize_factor, crop_bottom)
  X_test = pump_image_data(X_test_files, resize_factor, crop_bottom)

  y_train = np.asarray(y_train)
  y_val = np.asarray(y_val)
  y_test = np.asarray(y_test)

  print('X_train =', len(X_train_files))
  print('y_train =', len(y_train))
  print('X_val =', X_val.shape)
  print('y_val =', y_val.shape)
  print('X_test =', X_test.shape)
  print('y_test =', y_test.shape)

  # Make video of train/val/test
  # make_video(X_train_files, y_train, 'movie-train.mp4')
  # make_video(X_val_files, y_val, 'movie-val.mp4')
  # make_video(X_test_files, y_test, 'movie-test.mp4')



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
  image_gen = read_image_gen(data_gen, resize_factor, flip_images, flip_images_ratio = flip_images_ratio, crop_bottom = crop_bottom)

  print('Creating model.')
  model = create_model(model_type, resize_factor, crop_bottom)
  model.summary()

  if restore_weights:
    print('Restoring weights from ', restore_weights)
    model.load_weights(restore_weights)

  print('Train model ...')

  if not DEBUG:
    samples_per_epoch = len(X_train_files) * 1.33 if flip_images else len(X_train_files)
    model = train_model_on_gen(model, image_gen,
                        validation_data = (X_val, y_val),
                        lr = lr,
                        samples_per_epoch = samples_per_epoch,
                        nb_epoch = nb_epoch,
                        batch_size = batch_size)
  else:
    # DEBUG
    print('pump resize_factor = ', resize_factor)
    X_train = pump_image_data(X_train_files, resize_factor, crop_bottom)
    if flip_images:
      X_train, y_train = extend_with_flipped(X_train, y_train)
      # X_val, y_val = extend_with_flipped(X_val, y_val)

    # print('X_train =', X_train)
    # print('y_train =', y_train)
    model = train_model(model, X_train, y_train,
                        validation_data = (X_val, y_val),
                        lr = lr,
                        nb_epoch = nb_epoch,
                        batch_size = batch_size)

  print('Saving model to ', save_file)
  save_model(model, save_file)

  print('Inference on train data ...')

  sample = pump_image_data(X_train_files[:100] if len(X_train_files) > 100 else X_train_files, resize_factor, crop_bottom)
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
  # X_test = X_test[:100] if len(X_test) > 100 else X_test
  # y_test = y_test[:100] if len(y_test) > 100 else y_test

  print('Evaluate model on test data batch.')
  test_predicts = model.predict(X_test)
  rmse = np.sqrt(np.mean((test_predicts-y_test)**2))
  print("Test model evaluated RMSE:", rmse)

  pic_name = "%s_%s_test.png" % (model_type, save_time)
  y_test = y_test[:100] if len(y_test) > 100 else y_test
  test_predicts = test_predicts[:100] if len(test_predicts) > 100 else test_predicts
  make_fig(model_type, rmse, pic_name, y_test, test_predicts)


if __name__ == '__main__':
  main()
