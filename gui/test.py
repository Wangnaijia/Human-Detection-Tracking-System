# -*- coding: utf-8 -*-
from __future__ import print_function  # 确保代码同时在Python2.7和Python3上兼容
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import imutils

# 初始化我们的行人检测器
hog = cv2.HOGDescriptor()  # 初始化方向梯度直方图描述子
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # 设置支持向量机(Support Vector Machine)使得它成为一个预先训练好了的行人检测器

def test(img):
	img = imutils.resize(img, width=min(800, img.shape[1]))
	rects, scores = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)

	sc = [score[0] for score in scores]
	sc = np.array(sc)

	# 转换下输出格式(x,y,w,h) -> (x1,y1,x2,y2)
	for i in range(len(rects)):
		r = rects[i]
		rects[i][2] = r[0] + r[2]
		rects[i][3] = r[1] + r[3]

	# 非极大值移植
	pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)

	# 画出矩形框
	for (x, y, xx, yy) in pick:
		cv2.rectangle(img, (x, y), (xx, yy), (0, 0, 255), 2)

	cv2.imshow('a', img)
	cv2.waitKey(10)

def run(path):
	cap = cv2.VideoCapture(path)
	while (cap.isOpened()):
		ret, image = cap.read()
		if ret == 0:
			break
		test(image)

	cap.release()
	cv2.destroyAllWindows()
