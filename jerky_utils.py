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
jerky_sections['train5-complete'] = [
  [0, 117],
  [123, 128],
  [142, 151],
  [162, 178],
  [223, 247],
  [285, 297],
  [327, 381]
]
jerky_sections['train6-complete'] = [
  [0, 37],
  [95, 104],
  [141, 152],
  [160, 178],
  [201, 212],
  [251, 261],
  [311, 321],
  [366, 372],
  [396, 400],
  [487, 498],
  [546, 555],
  [565, 572],
  [603, 618],
  [678, 691],
  [706, 716],
  [778, 784],
  [820, 829],
  [873, 880],
  [932, 964],
  [1024, 1040],
  [1107, 1122],
  [1140, 1160],
  [1260, 1278],
  [1323, 1333],
  [1384, 1387],
  [1490, 1499],
  [1532, 1547],
  [1619, 1629],
  [1670, 1689],
  [1709, 1721],
  [1781, 1796],
  [1981, 1986],
  [2053, 2076],
  [2114, 2116],
  [2141, 2157],
  [2240, 2249],
  [2279, 2290],
  [2308, 2323],
  [2345, 2362],
  [2392, 2413],
  [2430, 2440],
  [2456, 2471],
  [2485, 2500],
  [2523, 2537],
  [2555, 2560],
  [2575, 2618],
  [2712, 2725],
  [2769, 2775],
  [2791, 2809],
  [2855, 2858],
  [2894, 2904],
  [2952, 2961],
  [2975, 2986],
  [3054, 3072],
  [3084, 3094],
  [3106, 3113],
  [3151, 3160],
  [3179, 3189],
  [3244, 3256],
  [3282, 3288],
  [3318, 3324],
  [3349, 3360]
]
jerky_sections['train7-complete'] = [
  [0, 304],
  [351, 356],
  [392, 398],
  [446, 452],
  [542, 549],
  [565, 580],
  [600, 612],
  [639, 645],
  [669, 676],
  [693, 701],
  [709, 721],
  [733, 738],
  [760, 772],
  [788, 799],
  [808, 816],
  [831, 842],
  [849, 861],
  [866, 874],
  [886, 908],
  [950, 957],
  [994, 999],
  [1004, 1049]
]
jerky_sections['train8-complete'] = [
  [0, 76],
  [94, 116],
  [138, 149],
  [198, 203],
  [299, 310],
  [329, 347],
  [359, 364],
  [388, 395],
  [429, 452],
  [476, 488],
  [512, 540],
  [554, 568],
  [603, 613],
  [633, 646],
  [719, 726],
  [751, 755],
  [781, 800],
  [836, 851],
  [891, 897],
  [923, 927],
  [1055, 1068],
  [1097, 1104],
  [1130, 1136],
  [1145, 1154],
  [1203, 1211],
  [1235, 1252],
  [1268, 1280],
  [1289, 1302],
  [1322, 1337],
  [1361, 1375],
  [1422, 1432],
  [1458, 1465],
  [1495, 1504],
  [1546, 1554],
  [1575, 1584],
  [1636, 1640],
  [1695, 1711],
  [1762, 1787],
  [1836, 1845],
  [1890, 1900],
  [1931, 1963],
  [2004, 2033],
  [2057, 2080],
  [2106, 2125],
  [2146, 2157],
  [2230, 2244],
  [2279, 2289],
  [2311, 2331],
  [2350, 2362],
  [2436, 2458],
  [2502, 2519],
  [2531, 2596],
  [2652, 2665],
  [2697, 2709],
  [2719, 2728],
  [2749, 2760],
  [2767, 2772],
  [2785, 2795],
  [2802, 2811],
  [2817, 2823],
  [2839, 2850],
  [2860, 2865],
  [2875, 2888],
  [2897, 2906],
  [2913, 2921],
  [2928, 2935],
  [2976, 2981],
  [2989, 2996],
  [3010, 3013],
  [3021, 3023],
  [3030, 3033],
  [3066, 3077],
  [3086, 3097],
  [3119, 3126],
  [3152, 3161],
  [3167, 3170],
  [3212, 3224],
  [3245, 3253],
  [3262, 3270],
  [3282, 3295],
  [3379, 3390],
  [3398, 3407],
  [3416, 3430],
  [3439, 3452],
  [3465, 3480],
  [3503, 3516]
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
