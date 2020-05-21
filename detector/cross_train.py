# -*- coding:utf-8 -*-
from sklearn.externals import joblib
from detector_config import *
import os
import glob
import shutil

def hard_negative_validate(model_name, logger, path= '../data/features/neg1'):

    """
    hard negative validation and copy the examples to final path
    :param path: final neg path
    :return: none
    """
    clf = joblib.load(os.path.join(model_path, model_name))
    num = 0
    total = 0
    for feat_path in glob.glob(os.path.join(neg_feat_path, '*.feat')):
        total += 1
        fd = joblib.load(feat_path)
        fd = fd.reshape(1, -1)

        pred = clf.predict(fd)
        if pred == 1:
            num+=1
            try:
                shutil.copy(feat_path, path)
            except IOError as e:
                logger.info("Unable to copy file. %s" % e)
                exit(1)
    logger.info('hard examples copied to {}'.format(path))
