#-*- coding:utf-8 -*-
import numpy as np
import cv2

#双目相机参数
class stereoCameral(object):
	def __init__(self):

		#左相机内参
		self.cam_matrix_left = np.array([[478.6033,0.,313.5121],[0.,475.4737,210.7335],[0.,0.,1]])
		#右相机内参
		self.cam_matrix_right = np.array([[478.4986,0.,278.2930],[0.,475.6240,209.1344],[0.,0.,1]])

		#左右相机畸变系数：[k1,k2,p1,p2,k3]
		self.distortion_l = np.array([[0.0709,-0.0941,-0.0000,0.0000,0.0000]])
		self.distortion_r = np.array([[0.0838,-0.1189,-0.0000,0.0000,0.0000]])

		#旋转矩阵
		# om = np.array([-0.00320,-0.00163,-0.00069])
		# self.R = cv2.Rodrigues(om)[0] #使用Rodrigues变换将om变换为R
		self.R = np.array([[0.9999,0.,0.0104],[0,1.0000,-0.0053],[-0.0104,0.0053,0.9999]])
		#平移矩阵
		self.T = np.array([-69.4262,0.2119,-0.1376])

		self.size = (640, 480) # 图像尺寸

		# 进行立体更正
		R1, R2, P1, P2, self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(self.cam_matrix_left, self.distortion_l,
																		  self.cam_matrix_right, self.distortion_r, self.size, self.R,
																		  self.T)
		# 计算更正map
		self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(self.cam_matrix_left, self.distortion_l, R1, P1, self.size, cv2.CV_16SC2)
		self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(self.cam_matrix_right, self.distortion_r, R2, P2, self.size, cv2.CV_16SC2)