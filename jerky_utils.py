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
