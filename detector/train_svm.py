# -*- coding: utf-8 -*-


from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os
from detector_config import *
import numpy as np
import time

def train_svm(model_name,logger):
    """
    训练SVM分类器
    :param model_name: 输入模型名 XXX.model
    :param logger: 控制台日志
    :return: 无
    """
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    logger.info("Shapes of training feature is %s, total num of labels is %s" % (np.array(fds).shape, len(labels)))

    t0 = time.time()
    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        logger.info("Training a Linear SVM Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fd_path = os.path.join(model_path, model_name)
        joblib.dump(clf, fd_path)
        t1 = time.time()
        logger.info("Classifier saved to {}".format(model_path))
        logger.info("The cast of time is {} seconds".format(t1 - t0))
