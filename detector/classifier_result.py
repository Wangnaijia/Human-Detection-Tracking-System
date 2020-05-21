# -*- coding:utf-8 -*-
from sklearn.externals import joblib
import sys
from detector_config import *
import time
import os
import glob


def classifier_test(model_name, logger):
    """
    通用的分类器测试模块，结果保存在result_path中的output_classifier中
    :param model_name: 用于测试的分类器名
    :param logger: 日志信息打印模块
    :return: 无
    """
    logger.info('Testing accuracy of classifier!')

    clf = joblib.load(os.path.join(model_path, model_name))
    pos_detect = 0
    pos_total = 0
    neg_detect = 0
    neg_total = 0
    detection = []
    t0 = time.time()

    if os.path.exists(test_feat_path_pos):
        for feat_path in glob.glob(os.path.join(test_feat_path_pos, '*.feat')):
            fd = joblib.load(feat_path)
            fd = fd.reshape(1, -1)

            pred = clf.predict(fd)
            if pred == 1:
                if clf.decision_function(fd) > 0.5:
                    detection.append(clf.decision_function(fd))
                    pos_detect += 1

        logger.info('Positive test feature path is:{}'.format(test_feat_path_pos))
        pos = os.listdir(test_feat_path_pos)
        pos_total = len(pos)  # 正样本总数
        logger.info('Positive test samples number:{}'.format(pos_total))
    else:
        logger.info('Positive test feature path need to be created')
    num = 0
    if os.path.exists(test_feat_path_neg):
        for feat_path in glob.glob(os.path.join(test_feat_path_neg, '*.feat')):

            fd = joblib.load(feat_path)
            fd = fd.reshape(1, -1)

            pred = clf.predict(fd)
            if pred == 0:
                neg_detect += 1
        logger.info('Negative data path is:{}'.format(test_feat_path_neg))
        neg = os.listdir(test_feat_path_neg)
        neg_total = len(neg)  # 负样本总数
        logger.info('Negative samples number:{}'.format(neg_total))
    else:
        logger.info('Negative test feature path need to be created')

    t2 = time.time()
    t = t2 - t0
    # record the results
    logger.info('Testing result saving...')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    output_file = os.path.join(result_path, "output_classifier.txt")
    with open(output_file, 'w') as f:

        f.write('\n The pos detected:\n')
        f.write(str(pos_detect))
        f.write('\n The pos total:\n')
        f.write(str(pos_total))
        f.write('\n The neg detected:\n')
        f.write(str(neg_detect))
        f.write('\n The neg total:\n')
        f.write(str(neg_total))
        f.write('\n Time cost is:\n')
        f.write(str(t))
        f.close()
    logger.info('Done')


