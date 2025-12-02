import cv2
import numpy as np

def compute_lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

def region_stats(image_bgr, labels):
    lab = compute_lab(image_bgr)
    h, w = labels.shape
    n_sp = labels.max() + 1

    mean_lab = np.zeros((n_sp, 3), dtype=np.float32)
    counts = np.zeros(n_sp, dtype=np.int32)

    flat_labels = labels.ravel()
    flat_lab = lab.reshape(-1, 3)

    for i in range(flat_labels.size):
        sp = flat_labels[i]
        mean_lab[sp] += flat_lab[i]
        counts[sp] += 1

    counts[counts == 0] = 1
    mean_lab /= counts[:, None]

    global_mean = mean_lab.mean(axis=0)
    return global_mean, mean_lab

def assign_keypoints_to_superpixels(keypoints, labels):
    h, w = labels.shape
    n_sp = labels.max() + 1

    sp_to_kp = [[] for _ in range(n_sp)]

    for idx, kp in enumerate(keypoints):
        x, y = kp.pt
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            sp_id = labels[yi, xi]
            sp_to_kp[sp_id].append(idx)

    return sp_to_kp
