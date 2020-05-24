#-*- coding:utf-8 -*-
import numpy as np
import stereoconfig


def runStereoCam(cap,cv2):
    """
    一些设定的分辨率格式：1280*480，640*240等
    """
    cv2.namedWindow("left")
    cv2.namedWindow("right")
    cv2.namedWindow("depth")
    cv2.createTrackbar("num","depth",0,10,lambda x:None)
    cv2.createTrackbar("blockSize","depth",5,255,lambda x:None)
    i = 0
    config = stereoconfig.stereoCameral()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            i += 1
            width = frame.shape[1] // 2
            frame1 = frame[:, :width, :]
            frame2 = frame[:, width:, :]

            # 根据更正map对图片进行重构
            img1_rectified = cv2.remap(frame1, config.left_map1, config.left_map2, cv2.INTER_LINEAR)
            img2_rectified = cv2.remap(frame2, config.right_map1, config.right_map2, cv2.INTER_LINEAR)

            # 将图片置为灰度图，为StereoBM作准备
            imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

            # 两个trackbar用来调节不同的参数查看效果
            num = cv2.getTrackbarPos("num", "depth")
            blockSize = cv2.getTrackbarPos("blockSize", "depth")
            if blockSize % 2 == 0:
                blockSize += 1
            if blockSize < 5:
                blockSize = 5
            img_channels=3
            # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
            # stereo = cv2.StereoSGBM_create(numDisparities=16 * num, blockSize=blockSize)
            stereo = cv2.StereoSGBM_create(minDisparity=1,
										   numDisparities=64,
										   blockSize=blockSize,
										   P1=8 * img_channels * blockSize * blockSize,
										   P2=32 * img_channels * blockSize * blockSize,
										   disp12MaxDiff=-1,
										   preFilterCap=1,
										   uniquenessRatio=10,
										   speckleWindowSize=100,
										   speckleRange=100,
										   mode=cv2.STEREO_SGBM_MODE_HH)
            disparity = stereo.compute(imgL, imgR)

            disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 将图片扩展至3d空间中，其z方向的值则为当前的距离
            threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., config.Q)

            cv2.imshow("left", img1_rectified)
            cv2.imshow("right", img2_rectified)
            cv2.imshow("depth", disp)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            if key == ord("s"):
                cv2.imwrite("./snapshot/BM_depth.jpg", disp)
                cv2.imwrite("./snapshot/left_" + str(i) + ".jpg", frame[:, :width, :])
                cv2.imwrite("./snapshot/right_" + str(i) + ".jpg", frame[:, width:, :])
                print("picture ", i, " saved.")
    cap.release()
    cv2.destroyAllWindows()


