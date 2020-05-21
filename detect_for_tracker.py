# -*- coding: utf-8 -*-

import numpy as np
import cv2, time
# from nms import py_cpu_nms
from imutils.object_detection import non_max_suppression
import imutils

# 初始化我们的行人检测器
hog = cv2.HOGDescriptor()  # 初始化方向梯度直方图描述子
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # 设置支持向量机(Support Vector Machine)使得它成为一个预先训练好了的行人检测器

# 帧率统计
fps = 0
fps_result = []
# 检测人数统计
num = 0
detect_num = []
# 帧数
frames = 0
t1 = time.time()

def detect_with_rects(img):

	img = imutils.resize(img, width=min(800, img.shape[1]))
	num = 0
	rects, scores = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
	# 转换下输出格式(x,y,w,h) -> (x1,y1,x2,y2)
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	# 非极大值移植
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)
	# 画出矩形框
	for (x, y, xx, yy) in pick:
		cv2.rectangle(img, (x, y), (xx, yy), (0, 255, 255), 2)
		num += 1
	return pick, img, num
	# cv2.imshow("test",img)
	# k = cv2.waitKey(1)
	# if k == 27:
	# 	return
	# print(fps)

def run(path):
	cap = cv2.VideoCapture(path)
	while (cap.isOpened()):
		ret, image = cap.read()
		if ret == 0:
			break
		test(image)

	cap.release()
	cv2.destroyAllWindows()


if __name__=='__main__':
	srcTest = '/home/wnj/projects/videos/007.avi'
	cap = cv2.VideoCapture(srcTest)
	while (cap.isOpened()):
		ret, image = cap.read()
		if ret == 0:
			break
		test(image)

	cap.release()
	cv2.destroyAllWindows()