# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
from detect_for_tracker import detect_with_rects
import imutils
import logging
sys.path.append('./')

from trackers.eco import ECOTracker

from imutils.object_detection import non_max_suppression

def logger_init():
    '''
    自定义python的日志信息打印配置
    :return logger: 日志信息打印模块
    '''
    # 获取logger实例，如果参数为空则返回root logger
    logger = logging.getLogger("PedestrianDetect")
    # 指定logger输出格式
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    # 文件日志
    # file_handler = logging.FileHandler("test.log")
    # file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter  # 也可以直接给formatter赋值
    # 为logger添加的日志处理器
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # 指定日志的最低输出级别，默认为WARN级别
    logger.setLevel(logging.INFO)
    return logger

def hog_detect(frame):
    '''
    :param frame: 初始帧
    :param model_path: 用于HOGDescriptor的SVM检测器
    :return: 检测到的最中心的人体目标bbox
    '''

    # opencv提取hog特征
    hog = cv2.HOGDescriptor()
    # 目标数
    detect_num = []
    fps = 0
    bbox = (0, 0, 0, 0)
    # opencv自带的训练好了的分类器
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # 构造了一个尺度scale=1.05的图像金字塔
    rects, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    height, width = frame.shape[:2]

    center = width
    num = len(rects)
    detect_num.append(num)
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    return pick

def ecotrack(frame, idx,tracker, height, width):

    timer = cv2.getTickCount()
    # 用后续帧更新tracker
    bbox = tracker.track(frame, True, True)

    # bbox xmin ymin xmax ymax
    frame = frame.squeeze()
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = cv2.rectangle(frame,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          (0, 255, 255),
                          1)

    score = tracker.score
    size = tuple(tracker.crop_size.astype(np.int32))
    score = cv2.resize(score, size)
    score -= score.min()
    score /= score.max()
    score = (score * 255).astype(np.uint8)
    # score = 255 - score
    score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
    pos = tracker._pos
    pos = (int(pos[0]), int(pos[1]))
    xmin = pos[1] - size[1] // 2
    xmax = pos[1] + size[1] // 2 + size[1] % 2
    ymin = pos[0] - size[0] // 2
    ymax = pos[0] + size[0] // 2 + size[0] % 2
    left = abs(xmin) if xmin < 0 else 0
    xmin = 0 if xmin < 0 else xmin
    right = width - xmax
    xmax = width if right < 0 else xmax
    right = size[1] + right if right < 0 else size[1]
    top = abs(ymin) if ymin < 0 else 0
    ymin = 0 if ymin < 0 else ymin
    down = height - ymax
    ymax = height if down < 0 else ymax
    down = size[0] + down if down < 0 else size[0]
    score = score[top:down, left:right]
    crop_img = frame[ymin:ymax, xmin:xmax]
    # if crop_img.shape != score.shape:
    #     print(xmin, ymin, xmax, ymax)
    score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
    frame[ymin:ymax, xmin:xmax] = score_map

    # counting FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # print idx
    frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
    # print("FPS:", str(int(fps)))
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    frame = imutils.resize(frame, width=min(800, frame.shape[1]))

    return frame, bbox, fps

def encode_region(region):
    region = (region[0], region[1],
            region[2] - region[0], region[3] - region[1])
    # output as integer
    return '{:.0f} {:.0f} {:.0f} {:.0f}'.format(region[0], region[1], region[2], region[3])


def run(path):
    bbox = []

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    height, width = frame.shape[:2]
    center = width
    if len(frame.shape) == 3:
        is_color = True
    else:
        is_color = False
        frame = frame[:, :, np.newaxis]
    # starting tracking
    tracker = ECOTracker(is_color)
    #
    while (True):
        ok, frame = cap.read()
        pick=hog_detect(frame)
        # check the first frame, if no object, do the next frame
        for (x, y, w, h) in pick:
            if abs(x + w / 2 - width / 2) + abs(y + h / 2 - height / 2) < center:
                center = abs(x + w / 2 - width / 2) + abs(y + h / 2 - height / 2)
                bbox = (x, y, w, h)
        if bbox:
            # initialize the tracker with frame[0] and bbox
            tracker.init(frame, bbox)
            break
        cv2.imshow('detect', frame)
        c = cv2.waitKey(10) & 0xff
        if c == 27:
            break
    #print(bbox)
    title = "ECO"
    idx = 0
    fps_result = []
    while (True):
        ret, image = cap.read()
        if ret == 0:
            break
        frame, bbox, fps = ecotrack(image, idx, tracker, height, width)
        bbox = encode_region(bbox)
        print bbox
        fps_result.append(fps)
        idx += 1
        cv2.imshow(title, frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    srcTest = '/home/wnj/projects/videos/003.avi'

    run(srcTest)



