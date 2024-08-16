# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes to load and preprocess CT images in DeepLesion.
# --------------------------------------------------------

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes, binary_opening, binary_dilation
#import nibabel as nib
#from config import config, default

def load_prep_img(imname, slice_idx, scale=1.0, max_im_size=512):
    """load volume, windowing, interpolate multiple slices, clip black border, resize according to spacing"""
    im = cv2.imread(imname, cv2.IMREAD_UNCHANGED)
    
    #c = [0, im.shape[0]-1, 0, im.shape[1]-1]

    im_shape = im.shape[0:2]
    im_scale = float(scale) / float(np.min(im_shape))  # simple scaling

    max_shape = np.max(im_shape)*im_scale
    if max_shape > max_im_size:
        im_scale1 = float(max_im_size) / max_shape
        im_scale *= im_scale1

    if im_scale != 1:
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    return im, im_scale 