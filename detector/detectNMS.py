#-*- coding:utf-8 -*-
import numpy as np
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
import time
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
from detector_config import *
from skimage import color
import matplotlib.pyplot as plt 
import os 
import glob

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size 
    equal to 'window_size'. The first image returned top-left 
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.

    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window 
    step_size - incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def detector(im, model_name, ifSave, logger):

    im = imutils.resize(im, width=min(400, im.shape[1]))
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 1.25
    fps = 0
    #List to store the detections
    detections = []
    #The current scale of the image
    scale = 0
    t1 = time.time()
    if model_name:
    # 若使用自己的模型
        clf = joblib.load(os.path.join(model_path, model_name))
        for im_scaled in pyramid_gaussian(im, downscale = downscale):
            #The list contains detections at the current scale
            if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
                break
            for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    continue
                im_window = color.rgb2gray(im_window)
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=normalize)

                fd = fd.reshape(1, -1)
                pred = clf.predict(fd)

                if pred == 1:

                    if clf.decision_function(fd) > 0.5:
                        detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd),
                        int(min_wdw_sz[0] * (downscale**scale)),
                        int(min_wdw_sz[1] * (downscale**scale))))

            scale += 1

        for (x_tl, y_tl, _, w, h) in detections:
            cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)

        rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
        sc = [score[0] for (x, y, score, w, h) in detections]
        logger.info("sc: {}".format(sc))
        sc = np.array(sc)
    else:
        # 初始化我们的行人检测器
        hog = cv2.HOGDescriptor()  # 初始化方向梯度直方图描述子
        hog.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector())  # 设置支持向量机(Support Vector Machine)使得它成为一个预先训练好了的行人检测器
        (rects, weights) = hog.detectMultiScale(im, winStride=(4, 4), padding=(8, 8), scale=1.05)
        # 应用非极大抑制方法，通过设置一个阈值来抑制那些重叠的边框
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        sc = None

    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
    logger.info("pick num:{}".format(len(pick)) )

    for(xA, yA, xB, yB) in pick:
        cv2.rectangle(im, (xA, yA), (xB, yB), (0, 255, 0), 2)
    fps = (fps + (1. / (time.time() - t1))) / 2


    if ifSave:
        output_file = os.path.join(result_path, "detect_result.txt")
        with open(output_file, 'w') as f:
            for r in pick:
                f.write(str(r))
                f.write('\n')
            f.write('\n fps results:\n')
            f.write(str(fps))
    return pick

def test_folder(foldername,logger):

    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        detector(filename,logger)

