#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
import sys
from detector_config import *
import time
import logging
from skimage import color


def sliding_window(image, window_size, step_size):
    """
    This function returns a patch of the input 'image' of size
    equal to 'window_size'. The first image returned top-left
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.
    :param image: Input
    :param window_size: Size of sliding window
    :param step_size: incremented size of window
    :return: a tuple (x, y, im_window)
    """
    for y in xrange(0,image.shape[0], step_size[1]):
        for x in xrange(0,image.shape[1], step_size[0]):
            yield (x, y, image[y: y+window_size[1], x:x+window_size[0]])

def sliding(im):
    """
    高斯滑窗，在各尺度上提取hog特征，若恰为min_wdw_sz，则不缩放
    :param filename: image
    :return: feature
    """
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 1.25
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # The list contains detections at the current scale
        if im_scaled.shape[0]<min_wdw_sz[1] or im_scaled.shape[1]<min_wdw_sz[0]:
            break
        for (x,y,im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # get feature of test
            im_window = color.rgb2gray(im_window)
            fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize=visualize,
                     transform_sqrt=normalize)
    return fd

    
def extract_features(logger):
    des_type = 'HOG'

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_path):
        os.makedirs(pos_feat_path)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_path):
        os.makedirs(neg_feat_path)

    # If feature directories don't exist, create them
    if not os.path.isdir(test_feat_path_neg):
        os.makedirs(test_feat_path_neg)

    logger.info("Calculating the descriptors for the positive samples and saving")

    t0 = time.time()
    num = 0

    for im_path in glob.glob(os.path.join(pos_im_path, "*")):

        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=normalize)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_path, fd_name)
        joblib.dump(fd, fd_path)
        num += 1
    logger.info("Positive features %d saved in %s." % (num, pos_feat_path))

    num = 0
    logger.info("Calculating the descriptors for the negative samples and saving")
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=normalize)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_path, fd_name)
        joblib.dump(fd, fd_path)
        num += 1
    logger.info("Negative features %d saved in %s." % (num, neg_feat_path))

    num = 0
    logger.info("Calculating the descriptors for test samples and saving")
    for im_path in glob.glob(os.path.join(test_im_path, "*")):
        im = imread(im_path, as_grey=True)
        fd = sliding(im)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(test_feat_path_neg, fd_name)
        joblib.dump(fd, fd_path)
        num += 1
    logger.info("Test features %d saved in %s." % (num, test_feat_path_neg))

    t1 = time.time()
    logger.info("Completed calculating in %f seconds" % (t1-t0))
    # pca.fit(feature_list)



