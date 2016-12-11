import numpy as np
import csv
import os
from PIL import Image

def read_signnames(file):
    with open(file) as f:
        csvreader = csv.reader(f)
        header = next(csvreader)
        signnames = {}
        for row in csvreader:
            signnames[int(row[0])] = row[1]
        return signnames

# Normalize data
def normalize(data):
    pixel_depth = 255
    return (data - pixel_depth / 2) / pixel_depth

# Read Data for Behavioral Clonning project
def bc_read_data(data_folder):
  driving_log_file = os.path.join(data_folder, 'driving_log.csv')
  driving_imgs_dir = os.path.join(data_folder, 'IMG')

  # print('Driving_log_file =', driving_log_file)
  # print('Driving_imgs_dir =', driving_imgs_dir)

  X_data = []
  y_data = []

  # Read
  with open(driving_log_file) as f:
    csvreader = csv.reader(f)
    for i, row in enumerate(csvreader):
      fname = os.path.basename(row[0])
      fname = os.path.join(driving_imgs_dir, fname)
      angle = float(row[3])
      X_data.append(fname)
      y_data.append(angle)
      # print('%d %s : %f' % (i, fname, angle))
      # if i > 20: break

  return X_data, np.asarray(y_data)


def read_data_gen(data, labels, batch_size=64):
  size = len(data)
  begin = 0
  # for begin in range(0, size, batch_size):
  while True:
    end = begin + batch_size
    if begin >= size:
        begin = 0
        end = batch_size
    if end > size:
        end = size
    # print("gen = %d : %d" % (begin, end))
    yield data[begin:end], labels[begin:end]
    begin += batch_size


def pump_image_data(data):
  data_img = [np.asarray(Image.open(img_file)) for img_file in data]
  data_img = normalize(np.asarray(data_img))
  return data_img


def read_image_gen(data_gen):
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
