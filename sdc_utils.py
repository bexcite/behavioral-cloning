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


def read_data_gen(data, labels, batch_size=64, all_data = None, attention = None, small_prob_tr = 1, small_tr = 0.1):
  if attention is not None:
    attention = np.asarray(attention)
  size = len(data)
  begin = 0
  batch_data = ['' for i in range(batch_size)]
  batch_labels = [0 for i in range(batch_size)]
  # for begin in range(0, size, batch_size):
  while True:

    for i in range(batch_size):
      ridx = np.random.randint(0, size)
      d = data[ridx]
      y = labels[ridx]

      # print('y = ', y)
      if abs(y) < small_tr:
        # print('small')
        keep_small_prob = np.random.uniform()
        # print('keep_small_prob = ', keep_small_prob)
        if keep_small_prob < small_prob_tr:
          # print('change')
          while abs(y) < small_tr:
            ridx = np.random.randint(0, size)
            d = data[ridx]
            y = labels[ridx]
            # print('change y=', y)

      '''
      if all_data is not None and attention is not None:
        att_prob = np.random.uniform()
        # print('att_prob =', att_prob)
        if att_prob < 0.1:
          att = np.random.choice(attention)
          # print('Apply attention mechanism %.3f - [%d]' % (att_prob, att))
          d = all_data[0][att]
          y = all_data[1][att]
      '''

      batch_data[i] = d
      batch_labels[i] = y


    # print('batch_data =', batch_data)
    # print('batch_labels =', batch_labels)
    yield batch_data, batch_labels

    #
    #
    # begin = np.random.randint(0, size)
    # end = begin + batch_size
    # if begin >= size:
    #     begin = 0
    #     end = batch_size
    # if end > size:
    #     end = size
    # # print("gen = %d : %d" % (begin, end))
    # yield data[begin:end], labels[begin:end]
    # begin += batch_size


def resize_image(img, resize_factor):
  im_shape = img.shape
  # img = imresize(img, 1.0 / resize_factor)
  if resize_factor > 1.0:
    size = int(im_shape[1] // resize_factor)
    img = imresize(img, (size, size))
    # print('shape =', img.shape)
  return img

def pump_image_data(data, resize_factor = 1.0, crop_bottom = None, norm = False):
  # print('Pump image data!!!')
  data_img = []
  for img_file in data:
    img = np.asarray(Image.open(img_file))

    # print('img = ', img)

    # Crop bottom part (remove car)
    if crop_bottom:
      img = img[:-crop_bottom,:]
      # print('img = : ', img.shape)

    # img = imresize(img, 1.0 / resize_factor)
    img = resize_image(img, resize_factor)

    if norm:
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

def read_image_gen(data_gen, resize_factor = 1.0, flip_images = 0.0, crop_bottom = None, augment = 0.0):
  for X_batch_files, y_batch in data_gen:
    # X_image = [np.asarray(Image.open(img_file)) for img_file in X_batch]
    X_image = pump_image_data(X_batch_files, 1.0, crop_bottom)
    # y_image = y_batch
    # X_image = np.asarray(X_image)
    y_image = np.asarray(y_batch)

    if flip_images > 0:
      X_image, y_image = extend_with_flipped(X_image, y_image, ratio=flip_images)
      # X_image_flipped = np.fliplr(np.copy(X_image))
      # y_image_flipped = -1.0 * np.copy(y_image)
      # X_image.append(X_image_flipped)
      # y_image.append(y_image_flipped)
      # print('flipped: %d : %d' % (len(X_image), len(y_image)))


    # shape = X_image.shape
    # print('shape =', shape)
    # x_batch = np.zeros((shape[0], shape[1] // resize_factor, shape[2] // resize_factor, shape[3]))
    x_batch = []
    y_batch = []
    # print('x_batch.shape =', x_batch.shape)

    for i in range(len(X_image)):
      data_image = X_image[i]
      data_label = y_image[i]

      data_orig_image = resize_image(data_image, resize_factor)
      x_batch.append(data_orig_image)
      y_batch.append(data_label)
      # y_image[i] = data_label

      if augment > 0:
        augment_prob = np.random.uniform()
        if augment_prob < augment:
          data_aug_image, data_aug_label = random_image_transform(data_image, data_label)
          data_aug_image = resize_image(data_aug_image, resize_factor)
          x_batch.append(data_aug_image)
          y_batch.append(data_aug_label)
      # Resize + Normalize

      # data_image = normalize(data_image)
      # x_batch[i, :, :, :] = data_image


    x_batch = np.asarray(x_batch)
    x_batch = normalize(x_batch)

    y_batch = np.asarray(y_batch)

    # print('x_batch.shape =', x_batch.shape)
    # print('y_batch.shape =', y_batch.shape)

    # print('im =', X_image[0].shape)
    # print('im2 =', X_image[0])
    # for img_file in X_batch:
    #   im = Image.open(img_file)
    #   im_as_array = np.asarray(im)
    #   print('im =', im_as_array.shape)
    # print('%d : %d' % (len(X_batch), len(y_batch)))
    # print(X_batch[:3])
    # print(y_batch[:3])
    # yield X_image, y_image
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
      '''
      X_data_files.extend(X_left_files)
      X_data_files.extend(X_right_files)
      y_left_data = np.copy(y_data) + np.absolute(y_data * 0.75)
      y_right_data = np.copy(y_data) - np.absolute(y_data * 0.75)
      y_data = np.hstack((y_data, y_left_data, y_right_data))
      '''
      for ldata, rdata, angle in zip(X_left_files, X_right_files, y_data):
        if angle == 0:
          y_left_data.append(0.10)
          y_right_data.append(-0.10)
        else:
          # l_angle = clip_angle(angle + abs(0.75 * angle))
          # r_angle = clip_angle(angle - abs(0.75 * angle))
          l_angle = clip_angle(angle + 0.10)
          r_angle = clip_angle(angle - 0.10)
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
        # print('ldata =', ldata)
        # print('rdata =', rdata)
        # print('angle =', angle)

    print('len X_data_files =', len(X_data_files))
    print('len y_data =', len(y_data))

    return X_data_files, y_data


def random_rotate(img):
    angle = np.random.randint(-25, 25)
#     print('angle =', angle)
    return ndimage.rotate(img, angle, reshape=False)

def random_gaussian_filter(img):
    sigma = np.random.choice([0.2,0.5,0.7,1])
#     sigma = 0.5
#     print('sigma =', sigma)
    return ndimage.gaussian_filter(img, sigma=sigma)

def random_noise(img):
    noise = np.zeros_like(img)
    noise_lvl = np.random.choice([10, 20, 30])
    cv2.randn(noise, 0, noise_lvl.flatten())
#     print("noise =", noise)
    return img + noise

def random_brightness(img):
    # based on https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.rp2xqyfig
    image1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2] * random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_trans(img, steer, trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x/trans_range * 2 * .1 # 160px = 0.1 angle
    tr_y = 40 * np.random.uniform() - 40 / 2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(img, Trans_M, (img.shape[1], img.shape[0]))

    return image_tr, steer_ang


def random_image_transform(img, angle):

    n = np.random.randint(4)
    # print('Rand transform ... n = ', n)
    if n == 0:
      # print('gaussian')
      img = random_gaussian_filter(img)
    elif n == 1:
      # print('noise')
      img = random_noise(img)
    elif n == 2:
      # print('brightness')
      img = random_brightness(img)
    elif n == 3:
      # print('trans')
      img, angle = random_trans(img, angle, 20)

    # transforms = {
    #     0: random_rotate,
    #     1: random_gaussian_filter,
    #     2: random_noise
    # }
    #
    # im = transforms[n](img)
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return img, angle
