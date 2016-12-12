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
jerky_sections['train9-complete'] = [
  [0, 52],
  [64, 72],
  [100, 109],
  [123, 133],
  [160, 173],
  [199, 213],
  [231, 246],
  [264, 289],
  [350, 357],
  [378, 391],
  [421, 432],
  [455, 475],
  [537, 552],
  [607, 620],
  [642, 654],
  [672, 677],
  [689, 705],
  [713, 724],
  [753, 760],
  [773, 780],
  [792, 802],
  [817, 831],
  [868, 881],
  [911, 926],
  [953, 971],
  [994, 1011],
  [1024, 1033],
  [1050, 1063],
  [1076, 1089],
  [1096, 1104],
  [1111, 1125],
  [1132, 1138],
  [1172, 1192],
  [1214, 1227],
  [1282, 1302],
  [1346, 1352],
  [1376, 1386],
  [1420, 1427],
  [1451, 1483],
  [1491, 1502],
  [1544, 1555],
  [1566, 1577],
  [1636, 1650],
  [1668, 1693],
  [1708, 1724],
  [1736, 1752],
  [1780, 1805],
  [1846, 1861],
  [1928, 1938],
  [1955, 1968],
  [2039, 2060],
  [2068, 2076],
  [2142, 2158],
  [2196, 2205],
  [2232, 2251],
  [2283, 2295],
  [2326, 2345],
  [2395, 2409],
  [2430, 2445],
  [2462, 2484],
  [2498, 2524],
  [2541, 2554],
  [2560, 2573],
  [2599, 2614],
  [2625, 2638],
  [2651, 2661],
  [2689, 2705],
  [2721, 2748],
  [2766, 2777],
  [2819, 2824],
  [2885, 2908],
  [2946, 2966],
  [2999, 3007],
  [3043, 3055],
  [3072, 3086],
  [3117, 3130],
  [3140, 3157],
  [3169, 3182],
  [3206, 3223],
  [3241, 3255],
  [3288, 3312],
  [3327, 3349],
  [3361, 3376],
  [3416, 3435],
  [3455, 3461],
  [3488, 3507],
  [3576, 3589],
  [3653, 3694]
]
jerky_sections['train10-complete'] = [
  [0, 61],
  [85, 95],
  [117, 132],
  [151, 175],
  [190, 206],
  [222, 232],
  [270, 297],
  [312, 325],
  [360, 372],
  [380, 383],
  [394, 401],
  [409, 416],
  [442, 458],
  [540, 555],
  [594, 599],
  [609, 617],
  [627, 639],
  [658, 705]
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
