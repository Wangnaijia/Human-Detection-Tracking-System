#-*- coding:utf-8 -*-
from .picture_cut import picCrop
from .extract_features import extract_features
from .train_svm import train_svm
from .classifier_result import classifier_test
from .PSO_PCA import PSO_PCA
from .cross_train import hard_negative_validate
from detectNMS import detector
from detector_config import *
import logging
import sys
import cv2

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


def main():
    logger = logger_init()

    filename = './expr/airport1.jpeg'
    frame = cv2.imread(filename)
    height, width = frame.shape[:2]
    center = width

    if flagCrop:
        num_crop = picCrop('./negRaw')
        logger.info('Croping num is:{}'.format(num_crop))
    if flagExtract:
        extract_features(logger)
    if flagPSO:
        PSO_PCA(logger)
    if flagTrain:
        train_svm('svm_test.model', logger)
    if flagHardValidate:
        hard_negative_validate('svm_test.model', logger)
    if flagClassifierTest:
        classifier_test('svm_cross.model', logger)

    pick = detector(frame, None, True, logger)



if __name__ == '__main__':
    main()