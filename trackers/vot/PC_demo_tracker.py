# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys

import detector
import logging
from KCFtracker import KCFtracker
from Sequence import Sequence
from tools import Tracking
sys.path.append('./')

from trackers.eco import ECOTracker


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


def runTrackers(trk_type):
    logger = logger_init()
    # load videos
    camUse = False
    selectROI = True
    bbox = []
    trk_types = ['ECO','KCF','DSST']
    if trk_type not in trk_types:
        logger.info('Please choose in ECO\KCF\DSST')
        return
    if camUse:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture("/home/wnj/projects/videos/people.mp4")
    ok, frame = video.read()
    height, width = frame.shape[:2]
    center = width
    if len(frame.shape) == 3:
        is_color = True
    else:
        is_color = False
        frame = frame[:, :, np.newaxis]
    # 手动绘框
    if selectROI:
        bbox = cv2.selectROI(frame, False)

    if trk_type == 'ECO' or 'KCF':
        while(True):
            if bbox:
                # initialize the tracker with frame[0] and bbox
                if trk_type == 'ECO':
                    tracker = ECOTracker(is_color)
                    tracker.init(frame, bbox)
                else:
                    tracker = KCFtracker(frame, bbox)
                bbox = (bbox[0] - 1, bbox[1] - 1,
                        bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2],bbox[3]), (0,255,0), thickness=2)
                break
            else:
                pick = detector.detector(frame, None, 0, logger)

                # check the first frame, if no object, do the next frame
                for (x, y, w, h) in pick:
                    if abs(x + w / 2 - width / 2) + abs(y + h / 2 - height / 2) < center:
                        center = abs(x + w / 2 - width / 2) + abs(y + h / 2 - height / 2)
                        bbox = (x, y, w, h)
        title = trk_type
        vis = False
        idx = 0
        while True:
            # 读新的帧
            ok, frame = video.read()
            if not ok:
                break
            idx += 1
            # start timer
            timer = cv2.getTickCount()
            # 用后续帧更新tracker
            bbox = tracker.track(frame)
            # bbox xmin ymin xmax ymax
            frame = frame.squeeze()
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
            frame = cv2.rectangle(frame,
                                  (int(bbox[0]), int(bbox[1])),
                                  (int(bbox[2]), int(bbox[3])),
                                  (0, 255, 255),
                                  1)
            if trk_type == 'ECO':
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
            cv2.putText(frame, "tracker : " + trk_type, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            # img_writer.write(frame)
            cv2.imshow(title, frame)

            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                break
            if k == ord(' '):
                cv2.waitKey(0)

        tracker.quit()
    elif trk_type == 'DSST':
        sequence = Sequence(path='/home/wnj/projects/Visual_Tracking_api-master', name='bicycle',
                            region_format='rectangle')
        Tracking(sequence, tracker_list=['DSSTtracker'], visualize=False)




if __name__ == "__main__":

    runTrackers('KCF')
