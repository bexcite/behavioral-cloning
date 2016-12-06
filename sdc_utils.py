import numpy as np
import csv
import os

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

  return X_data, y_data
