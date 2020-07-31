# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import torch
import cv2
import numpy as np


def tens2image(tens):
    """Converts tensor with 2 or 3 dimensions to numpy array"""
    im = tens.numpy()

    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)

    if im.ndim == 3:
        im = im.transpose((1, 2, 0))

    return im


def pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def fixed_resize(sample, resolution, flagval=None):
    """
    Fixed resize to
    resolution (tuple): resize image to size specified by tuple eg. (512, 512).
    resolution (int): bring smaller side to resolution eg. image of shape 321 x 481 -> 512 x 767
    """
    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[int(np.argmax(sample.shape[:2]))] = int(
            round(float(resolution) / np.min(sample.shape[:2]) * np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample


def im_normalize(im, max_value=1):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value * (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()


def ind2sub(array_shape, inds):
    rows, cols = [], []
    for k in range(len(inds)):
        if inds[k] == 0:
            continue
        cols.append((inds[k].astype('int') // array_shape[1]))
        rows.append((inds[k].astype('int') % array_shape[1]))
    return rows, cols
