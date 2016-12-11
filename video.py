'''
Makes video from training raw dataset.

It helps to remove particular frames that we don't need during
the training like swirling or some `driver` errors.

'''

import cv2
import argparse
import numpy as np
import math
from moviepy.editor import ImageSequenceClip
from sdc_utils import bc_read_data, normalize

DEBUG = 0

def make_video(data_files, labels, output_file, fps = 10):

  h, w = 160, 320
  radius = 30

  def process_image(get_frame, t):
    idx = int(round(t * fps))
    if idx >= len(labels):
      return get_frame(t)
    angle = labels[idx]
    # print('t =', t, 'idx =', idx, ' y_data =', y_data[idx])
    image = get_frame(t)
    img = np.copy(image)

    # Cliping for square
    # cv2.rectangle(img, (w // 2 - w // 4, 0), (w // 2 + w // 4, h), [0, 0, 255], thickness = 1 )

    # Main circle
    cv2.circle(img, (w // 2, h), radius, [255, 255, 255], thickness = 5 )

    # Steering angle pointer
    cv2.circle(img, (w // 2 + round(radius * math.sin(angle)), h - round(radius * math.cos(angle))), 2, [0, 155, 0], thickness = 2 )

    # Frame number
    cv2.putText(img, "{}".format(idx), (w // 2 + 80, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
    return img

  if DEBUG:
    data_files = data_files[:100]

  clip = ImageSequenceClip(data_files, fps = fps)
  clip = clip.fl(process_image)
  clip.write_videofile(output_file)


def main():
  print("Hello there")

  # parse arguments
  parser = argparse.ArgumentParser(description="Making video from provided dataset folder")
  parser.add_argument('--dataset', type=str, help='dataset folder with csv file and image folders')
  parser.add_argument('--output', type=str, default='movie.mp4', help='output movie file')

  args = parser.parse_args()

  dataset_path = args.dataset
  output_file = args.output

  if not dataset_path:
    parser.error("No dataset is not specified")

  print('dataset = ', dataset_path)
  print('output = ', output_file)

  X_data_files, y_data = bc_read_data(dataset_path)

  print('len X_data_files =', len(X_data_files))
  print('len y_data =', len(y_data))

  print('Making video ...')
  make_video(X_data_files, y_data, output_file)


if __name__ == '__main__':
  main()
