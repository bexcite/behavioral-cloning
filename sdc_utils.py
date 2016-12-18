import numpy as np
import csv
import os
from PIL import Image
from jerky_utils import remove_jerky_sections
from scipy.misc import imresize

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

  X_center_data = []
  X_left_data = []
  X_right_data = []
  y_data = []

  # Read
  with open(driving_log_file) as f:
    csvreader = csv.reader(f)
    for i, row in enumerate(csvreader):

      cname = os.path.basename(row[0])
      cname = os.path.join(driving_imgs_dir, cname)
      X_center_data.append(cname)

      lname = os.path.basename(row[1])
      lname = os.path.join(driving_imgs_dir, lname)
      X_left_data.append(lname)

      rname = os.path.basename(row[2])
      rname = os.path.join(driving_imgs_dir, rname)
      X_right_data.append(rname)

      angle = float(row[3])

      # X_data.append(cname)
      # X_data.append(data_row)
      y_data.append(angle)
      # print('%d %s : %f' % (i, fname, angle))
      # if i > 20: break

  y_data = np.asarray(y_data)

  return X_center_data, X_left_data, X_right_data, y_data


def read_data_gen(data, labels, batch_size=64):
  size = len(data)
  begin = 0
  # for begin in range(0, size, batch_size):
  while True:
    begin = np.random.randint(0, size)
    end = begin + batch_size
    if begin >= size:
        begin = 0
        end = batch_size
    if end > size:
        end = size
    # print("gen = %d : %d" % (begin, end))
    yield data[begin:end], labels[begin:end]
    begin += batch_size


def pump_image_data(data, resize_factor = 1.0):
  data_img = []
  for img_file in data:
    img = np.asarray(Image.open(img_file))
    img = imresize(img, 1.0 / resize_factor)
    # print('img = %f : ', (resize_factor, img.shape))
    img = normalize(img)
    data_img.append(img)
  # data_img = [np.asarray(Image.open(img_file)) for img_file in data]
  # data_img = normalize(np.asarray(data_img))

  return np.asarray(data_img)

def extend_with_flipped(data, labels, ratio=0.3):

  n = len(data)
  ridx = np.random.permutation(n)
  n_part = int(n*ratio)
  ridx = ridx[:n_part]

  # print('flip only ridx =', ridx)

  data_flipped = np.fliplr(np.copy(data[ridx]))
  labels_flipped = -1.0 * np.copy(labels[ridx])

  ndata = np.concatenate([data, data_flipped])
  nlabels = np.concatenate([labels, labels_flipped])
  return ndata, nlabels

def read_image_gen(data_gen, resize_factor = 1.0, flip_images = False, flip_images_ratio = 0.3):
  for X_batch_files, y_batch in data_gen:
    # X_image = [np.asarray(Image.open(img_file)) for img_file in X_batch]
    X_image = pump_image_data(X_batch_files, resize_factor)
    # y_image = y_batch
    # X_image = np.asarray(X_image)
    y_image = np.asarray(y_batch)

    if flip_images:
      X_image, y_image = extend_with_flipped(X_image, y_image, ratio=flip_images_ratio)
      # X_image_flipped = np.fliplr(np.copy(X_image))
      # y_image_flipped = -1.0 * np.copy(y_image)
      # X_image.append(X_image_flipped)
      # y_image.append(y_image_flipped)
      # print('flipped: %d : %d' % (len(X_image), len(y_image)))

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


def load_all_datasets(base_path, remove_jerky = False, left_right = False):
  datasets = [
    os.path.join(base_path, 'train1-complete'),
    os.path.join(base_path, 'train2-complete'),
    os.path.join(base_path, 'train3-complete'),
    os.path.join(base_path, 'train4-complete'),
    os.path.join(base_path, 'train5-complete'),
    os.path.join(base_path, 'train6-complete'),
    os.path.join(base_path, 'train7-complete'),
    os.path.join(base_path, 'train8-complete'),
    os.path.join(base_path, 'train9-complete'),
    os.path.join(base_path, 'train10-complete')
  ]

  X_all_data = []

  y_all_data = []

  for dataset_path in datasets:
    if os.path.isdir(dataset_path):
      X_data_files, y_data = load_dataset(dataset_path, remove_jerky, left_right)
      X_all_data.extend(X_data_files)
      y_all_data.extend(y_data)

  return X_all_data, y_all_data


def load_dataset(dataset_path, remove_jerky = False, left_right=False):
    X_center_files, X_left_files, X_right_files, y_data = bc_read_data(dataset_path)

    if remove_jerky:
      print('Remove jerky sections ...')
      X_center_files, X_left_files, X_right_files, y_data = remove_jerky_sections(
              X_center_files,
              X_left_files,
              X_right_files,
              y_data,
              dataset_path)

    print('len X_center_files =', len(X_center_files))
    print('len X_left_files =', len(X_left_files))
    print('len X_right_files =', len(X_center_files))
    # print('len y_data =', y_data)

    X_data_files = []
    X_data_files.extend(X_center_files)
    print('len X_data_files init =', len(X_data_files))

    if left_right:
      y_left_data = []
      y_right_data = []
      '''
      X_data_files.extend(X_left_files)
      X_data_files.extend(X_right_files)
      y_left_data = np.copy(y_data) + np.absolute(y_data * 0.75)
      y_right_data = np.copy(y_data) - np.absolute(y_data * 0.75)
      y_data = np.hstack((y_data, y_left_data, y_right_data))
      '''
      for ldata, rdata, angle in zip(X_left_files, X_right_files, y_data):
        if angle == 0:
          y_left_data.append(0.1)
          y_right_data.append(-0.1)
        else:
          y_left_data.append(angle + abs(0.75 * angle))
          y_right_data.append(angle - abs(0.75 * angle))

      X_data_files.extend(X_left_files)
      X_data_files.extend(X_right_files)

      y_left_data = np.asarray(y_left_data)
      y_right_data = np.asarray(y_right_data)
      print('len y_left_data =', len(y_left_data))
      print('len y_right_data =', len(y_right_data))
      y_data = np.hstack((y_data, y_left_data, y_right_data))

      print('len X_data_files =', len(X_data_files))
      print('len y_data =', len(y_data))
        # print('ldata =', ldata)
        # print('rdata =', rdata)
        # print('angle =', angle)

    print('len X_data_files =', len(X_data_files))
    print('len y_data =', len(y_data))

    return X_data_files, y_data
