import cv2
import numpy as np

def run_slic_opencv(
    image_bgr,
    region_size=25,
    ruler=10.0,
    num_iterations=10,
    algorithm="SLIC"
):
    alg_map = {
        "SLIC": cv2.ximgproc.SLIC,
        "SLICO": cv2.ximgproc.SLICO,
        "MSLIC": cv2.ximgproc.MSLIC,
    }
    algo = alg_map[algorithm]

    slic = cv2.ximgproc.createSuperpixelSLIC(
        image_bgr,_algorithm=algo,
        region_size=region_size,
        ruler=float(ruler)
    )
    slic.iterate(num_iterations)

    labels = slic.getLabels()
    n_superpixels = slic.getNumberOfSuperpixels()
    return labels, n_superpixels
