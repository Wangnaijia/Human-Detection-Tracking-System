#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
# 提取图片hog特征
from skimage.feature import hog
from skimage.io import imread
from skimage import data, exposure
import glob
import os

import time
import matplotlib.pyplot as plt
import cv2
import numpy as np

def hogShow(img):
    image = cv2.imread()
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title('Input image')
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()

if __name__=='__main__':
    hogShow()








