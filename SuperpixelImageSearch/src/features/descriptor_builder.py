import numpy as np
import cv2

from src.superpixels.slic_opencv import run_slic_opencv
from src.features.sift_orb import compute_sift, compute_orb, global_descriptor
from src.features.superpixel_region_features import (
    region_stats, assign_keypoints_to_superpixels
)

def build_descriptor(image_bgr, use_sift=True):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ---- Superpixels ----
    labels, _ = run_slic_opencv(image_bgr)

    # ---- Keypoints ----
    if use_sift:
        kps, desc = compute_sift(gray)
        d_dim = 128
    else:
        kps, desc = compute_orb(gray)
        d_dim = 32

    sift_global = global_descriptor(desc)

    # ---- Superpixel features ----
    global_lab, mean_lab_sp = region_stats(image_bgr, labels)

    # ---- Region-SIFT assignment ----
    sp_to_idx = assign_keypoints_to_superpixels(kps, labels)
    n_sp = labels.max() + 1

    region_sift = np.zeros((n_sp, d_dim), dtype=np.float32)
    for sp_id, idxs in enumerate(sp_to_idx):
        if len(idxs) > 0:
            region_sift[sp_id] = desc[idxs].mean(axis=0)
        else:
            region_sift[sp_id] = np.zeros((d_dim,), dtype=np.float32)

    global_region_sift = region_sift.mean(axis=0)

    # ---- Final descriptor ----
    descriptor = np.concatenate([
        sift_global,       # 128  
        global_lab,        # 3
        global_region_sift # 128
    ]).astype(np.float32)

    return descriptor
