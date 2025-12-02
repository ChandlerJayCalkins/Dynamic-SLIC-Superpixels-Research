import cv2
import numpy as np

def compute_sift(gray):
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        desc = np.zeros((0, 128), dtype=np.float32)
    return kps, desc

def compute_orb(gray, n_features=1000):
    orb = cv2.ORB_create(nfeatures=n_features)
    kps, desc = orb.detectAndCompute(gray, None)
    if desc is None:
        desc = np.zeros((0, 32), dtype=np.uint8)
    return kps, desc

def global_descriptor(desc):
    if desc.shape[0] == 0:
        return np.zeros((desc.shape[1],), dtype=np.float32)
    return desc.mean(axis=0).astype(np.float32)
