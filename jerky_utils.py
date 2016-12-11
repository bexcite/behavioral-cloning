import os
import numpy as np

# Mapped Jerky sections that we need to remove - depends on dataset
jerky_sections = {}
jerky_sections['train1-complete'] = [
  [0, 80],
  [295, 350],
  [429, 445],
  [540, 550],
  [680, 686],
  [718, 760],
  [773, 778],
  [1007, 1013],
  [1352, 1370],
  [1459, 1474],
  [1990, 2001],
  [2166, 2172],
  [2318, 2322]
]
jerky_sections['train2-complete'] = [
  [0, 30],
  [300, 320],
  [900, 965],
  [1546, 1575],
  [2020, 2040],
  [2170, 2200],
  [3470, 3538]
]
jerky_sections['train3-complete'] = [
  [157, 191],
  [396, 413],
  [495, 509],
  [702, 706],
  [848, 855],
  [1199, 1206],
  [1262, 1277],
  [1304, 1317],
  [1417, 1428],
  [1465, 1511],
  [1578, 1582],
  [1661, 1664],
  [1893, 1900],
  [1939, 1946],
  [2026, 2044],
  [2084, 2095],
  [2327, 2338],
  [2509, 2521],
  [2610, 2617],
  [2693, 2705],
  [2754, 2768],
  [2807, 2820],
  [2886, 2896],
  [2932, 2939],
  [3130, 3147],
  [3252, 3263],
  [3311, 3340],
  [3393, 3401],
  [3432, 3444],
  [3473, 3495],
  [3536, 3544],
  [3770, 3785],
  [3790, 3796],
  [3854, 3862],
  [3957, 3970],
  [4033, 4050],
  [4212, 4235],
  [4272, 4284],
  [4365, 4378]
]
jerky_sections['train4-complete'] = [
  [0, 79],
  [121, 134],
  [202, 207],
  [234, 246],
  [280, 291],
  [383, 392],
  [412, 416],
  [449, 457],
  [625, 639],
  [721, 727],
  [827, 840],
  [898, 912],
  [952, 966],
  [1014, 1029],
  [1100, 1113],
  [1153, 1161],
  [1218, 1235],
  [1346, 1363],
  [1453, 1468],
  [1560, 1585],
  [1702, 1721],
  [1731, 1757],
  [1794, 1817],
  [1857, 1872],
  [1924, 1936],
  [2175, 2183],
  [2265, 2277],
  [2307, 2330],
  [2403, 2411],
  [2459, 2477],
  [2505, 2525],
  [2576, 2600],
  [2640, 2656],
  [2870, 2885]
]



def remove_jerky_sections(data, labels_data, dataset_path):
  # Idxs to remove from dataset (bad driver:))

  dataset_name = os.path.basename(os.path.normpath(dataset_path))
  print('dataset_name =', dataset_name)

  sections_to_remove = jerky_sections.get(dataset_name, [])

  prev_size = len(data)

  def leave_elements_idx(n, to_remove):
    if len(to_remove) == 0: return np.arange(n)
    all_list = []
    for rm in to_remove:
        rm_arr = np.arange(rm[0], rm[1])
        all_list.append(rm_arr)
    conc = np.concatenate(all_list, axis = 0)
    return np.delete(np.arange(n), conc)

  leave_idx = leave_elements_idx(len(data), sections_to_remove)

  data_files = np.asarray(data)
  data_files = data_files[leave_idx]
  data_files = data_files.tolist()
  labels = labels_data[leave_idx]

  new_size = len(data_files)

  print('Removed %d frames from dataset %s' % (prev_size - new_size, dataset_name))

  return data_files, labels
