#-*- coding:utf-8 -*-
import random
import cv2
import numpy as np
import os
from PIL import Image


def picCrop(filePath, width=64, height=128):
    """

    :param filePath: 待处理的图片
    :param width: 裁剪成图的宽
    :param height: 裁剪成图的高
    :return: 执行次数
    """
    random.seed(1)
    num = 0
    for childDir in os.listdir(filePath):
        f_im = os.path.join(filePath,childDir)
        image = cv2.imread(f_im)

        # fileNames = np.array([[childDir]])
        for j in range(10):
            y = int(random.random() * (image.shape[0] - height))
            x = int(random.random() * (image.shape[1] - width))
            # box = (y, x, image.shape[1] - y - height, image.shape[0] - x - width)
            # region = image.crop(box)
            cropImg = image[y:y+128, x:x+64]
            newName = os.path.splitext(childDir)[0] + "_" + str(j) + os.path.splitext(childDir)[1]

            cv2.imwrite(newName, cropImg)
            num += 1

    return(num)



