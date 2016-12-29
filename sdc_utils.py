'''
Helper Functions

'''

import numpy as np
import csv
import os
from PIL import Image
from jerky_utils import remove_jerky_sections
from scipy.misc import imresize
from scipy import ndimage
import cv2

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

      y_data.append(angle)

  y_data = np.asarray(y_data)

  return X_center_data, X_left_data, X_right_data, y_data


def read_data_gen(data, labels, batch_size=64, all_data = None, attention = None, small_prob_tr = 1, small_tr = 0.1):
  if attention is not None:
    attention = np.asarray(attention)
  size = len(data)
  begin = 0
  batch_data = ['' for i in range(batch_size)]
  batch_labels = [0 for i in range(batch_size)]

  while True:

    for i in range(batch_size):
      ridx = np.random.randint(0, size)
      d = data[ridx]
      y = labels[ridx]

      if abs(y) < small_tr:
        keep_small_prob = np.random.uniform()
        if keep_small_prob < small_prob_tr:
          while abs(y) < small_tr:
            ridx = np.random.randint(0, size)
            d = data[ridx]
            y = labels[ridx]


      # It was used for experiments when you feed a particular section with
      # bigger probablity to appear in training set (like difficult turns)
      if all_data is not None and attention is not None:
        att_prob = np.random.uniform()
        # print('att_prob =', att_prob)
        if att_prob < 0.1:
          att = np.random.choice(attention)
          # print('Apply attention mechanism %.3f - [%d]' % (att_prob, att))
          d = all_data[0][att]
          y = all_data[1][att]


      batch_data[i] = d
      batch_labels[i] = y

    yield batch_data, batch_labels



def resize_image(img, resize_factor):
  im_shape = img.shape
  if resize_factor > 1.0:
    # Calc square size based on biggest side
    size = int(im_shape[1] // resize_factor)
    img = imresize(img, (size, size))
  return img

def pump_image_data(data, resize_factor = 1.0, crop_bottom = None, norm = False):
  data_img = []
  for img_file in data:
    img = np.asarray(Image.open(img_file))

    # Crop bottom part (remove car)
    if crop_bottom:
      img = img[:-crop_bottom,:]
      # print('img = : ', img.shape)

    img = resize_image(img, resize_factor)

    if norm:
      img = normalize(img)

    data_img.append(img)

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

def read_image_gen(data_gen, resize_factor = 1.0, flip_images = 0.0, crop_bottom = None, augment = 0.0):
  for X_batch_files, y_batch in data_gen:
    X_image = pump_image_data(X_batch_files, 1.0, crop_bottom)
    y_image = np.asarray(y_batch)

    if flip_images > 0:
      X_image, y_image = extend_with_flipped(X_image, y_image, ratio=flip_images)

    x_batch = []
    y_batch = []

    for i in range(len(X_image)):
      data_image = X_image[i]
      data_label = y_image[i]

      data_orig_image = resize_image(data_image, resize_factor)
      x_batch.append(data_orig_image)
      y_batch.append(data_label)

      if augment > 0:
        augment_prob = np.random.uniform()
        if augment_prob < augment:
          data_aug_image, data_aug_label = random_image_transform(data_image, data_label)
          data_aug_image = resize_image(data_aug_image, resize_factor)
          x_batch.append(data_aug_image)
          y_batch.append(data_aug_label)

    x_batch = np.asarray(x_batch)
    x_batch = normalize(x_batch)

    y_batch = np.asarray(y_batch)

    yield x_batch, y_batch

def load_datasets(base_path, datasets, remove_jerky = False, left_right = False):
  X_all_data = []
  y_all_data = []
  datasets_path = [os.path.join(base_path, p) for p in datasets]
  for dataset_path in datasets_path:
    if os.path.isdir(dataset_path):
      X_data_files, y_data = load_dataset(dataset_path, remove_jerky, left_right)
      X_all_data.extend(X_data_files)
      y_all_data.extend(y_data)

  return X_all_data, y_all_data


def load_all_datasets(base_path, remove_jerky = False, left_right = False):
  datasets = [
    'train1-complete',
    'train2-complete',
    'train3-complete',
    'train4-complete',
    'train5-complete',
    'train6-complete',
    'train7-complete',
    'train8-complete',
    'train9-complete',
    'train10-complete',
    'data'
  ]

  return load_datasets(base_path, datasets, remove_jerky = remove_jerky, left_right = left_right)

def clip_angle(ang):
  if ang < 0:
    return max(ang, -1.0)
  return min(ang, 1.0)

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
      for ldata, rdata, angle in zip(X_left_files, X_right_files, y_data):
        if angle == 0:
          y_left_data.append(0.1)
          y_right_data.append(-0.1)
        else:
          # l_angle = clip_angle(angle + abs(0.75 * angle))
          # r_angle = clip_angle(angle - abs(0.75 * angle))
          l_angle = clip_angle(angle + 0.1)
          r_angle = clip_angle(angle - 0.1)
          y_left_data.append(l_angle)
          y_right_data.append(r_angle)

      X_data_files.extend(X_left_files)
      X_data_files.extend(X_right_files)

      y_left_data = np.asarray(y_left_data)
      y_right_data = np.asarray(y_right_data)
      print('len y_left_data =', len(y_left_data))
      print('len y_right_data =', len(y_right_data))
      y_data = np.hstack((y_data, y_left_data, y_right_data))

      print('len X_data_files =', len(X_data_files))
      print('len y_data =', len(y_data))

    print('len X_data_files =', len(X_data_files))
    print('len y_data =', len(y_data))

    return X_data_files, y_data


def random_rotate(img):
    angle = np.random.randint(-25, 25)
    return ndimage.rotate(img, angle, reshape=False)

def random_gaussian_filter(img):
    sigma = np.random.choice([0.2,0.5,0.7,1])
    return ndimage.gaussian_filter(img, sigma=sigma)

def random_noise(img):
    noise = np.zeros_like(img)
    noise_lvl = np.random.choice([10, 20, 30])
    cv2.randn(noise, 0, noise_lvl.flatten())
    return img + noise

def random_brightness(img):
    # based on https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.rp2xqyfig
    image1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:,:,2] = image1[:,:,2] * random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_trans(img, steer, trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform() - trans_range / 2
    # steer_ang = steer + tr_x/trans_range * 2 * .1 # 160px = 0.1 angle (or 0.15)
    steer_ang = steer + tr_x/160.0 * 0.1 # 160px = 0.1 angle
    tr_y = 40 * np.random.uniform() - 40 / 2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(img, Trans_M, (img.shape[1], img.shape[0]))

    return image_tr, steer_ang


def random_image_transform(img, angle):
    n = np.random.randint(4)
    if n == 0:
      img = random_gaussian_filter(img)
    elif n == 1:
      img = random_noise(img)
    elif n == 2:
      img = random_brightness(img)
    elif n == 3:
      img, angle = random_trans(img, angle, 20)

    return img, angle
