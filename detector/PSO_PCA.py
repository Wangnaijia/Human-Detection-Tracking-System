from sklearn.externals import joblib
import os
import numpy as np
import glob
import random
import copy
import sys
from detector_config import *

sys.path.append('/home/wnj/projects/TrackingDemo/lib/libsvm/python')
from svmutil import svm_train

n = 2000
train_feat_path = './features/train'

birds = 20  # size of population
maxgen = 50
pos = []  # population of class
speed = []
bestpos = []
initpos = []
tempfit = []
birdsbestpos = []

dict_fds = []
labels = []
allbestpos = []
w = 1  # best belongs to [0.8,1.2]
c1 = 2
c2 = 2
r1 = random.uniform(0, 1)
r2 = random.uniform(0, 1)
m = 31.53



def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)
    #    joblib.dump(meanVal,'./features/PCA/meanVal_train_%s.mean' %m)
    newData = dataMat - meanVal
    return newData, meanVal


def pca(dataMat, n, logger):
    logger.info("Start to do PCA...")
    newData, meanVal = zeroMean(dataMat)

    eigVals = joblib.load('../data/PCA/eigVals_train_%s.eig' % m)
    eigVects = joblib.load('../data/PCA/eigVects_train_%s.eig' % m)

    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    #    joblib.dump(n_eigVect,'./features/PCA/n_eigVects_train_%s_%s.eig' %(m,n))
    lowDDataMat = newData * n_eigVect
    return lowDDataMat




def CalDis(list,fds):
    fitness = 0.0
    param = '-t 2 -v 3 -c %s -g %s' % (list[0], list[1])
    fitness = svm_train(labels, fds, param)
    return fitness

def FindBirdsMostPos(logger,fds):
    best = CalDis(bestpos[0],fds)
    index = 0
    for i in range(birds):
        logger.info("\n>>>>>The %d'd time to find globel best pos. Total %d times.\n" % (i + 1, birds))
        tempfit[i] = CalDis(bestpos[i],fds)
        if tempfit[i] > best:
            best = tempfit[i]
            index = i
            logger.info( '------- %d: %f' % (index, best))
    return best, bestpos[index]

def NumMulVec(num, list):  # result is in list
    for i in range(len(list)):
        list[i] *= num
    return list


def VecSubVec(list1, list2):  # result is in list1
    for i in range(len(list1)):
        list1[i] -= list2[i]
    return list1


def VecAddVec(list1, list2):  # result is in list1
    for i in range(len(list1)):
        list1[i] += list2[i]
    return list1


def UpdateSpeed():
    # global speed
    for i in range(birds):
        temp1 = NumMulVec(w, speed[i][:])
        temp2 = VecSubVec(bestpos[i][:], pos[i])
        temp2 = NumMulVec(c1 * r1, temp2[:])
        temp1 = VecAddVec(temp1[:], temp2)
        temp2 = VecSubVec(birdsbestpos[:], pos[i])
        temp2 = NumMulVec(c2 * r2, temp2[:])
        speed[i] = VecAddVec(temp1, temp2)


def UpdatePos(logger,fds):
    logger.info("Update Pos.")
    global bestpos, birdsbestpos, tempfit
    for i in range(birds):
        if pos[i][0] + speed[i][0] > 0 and pos[i][1] + speed[i][1] > 0:
            VecAddVec(pos[i], speed[i])
            if CalDis(pos[i],fds) > tempfit[i]:
                bestpos[i] = copy.deepcopy(pos[i])
    best_predict, birdsbestpos = FindBirdsMostPos(logger,fds)
    return best_predict, birdsbestpos

def PSO_PCA(logger):
    fds = []
    num = 0
    for feat_path in glob.glob(os.path.join(pos_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)
        num += 1
    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
        num += 1
    fds = np.array(fds, dtype=float)
    fds.shape = num, -1
    fds = pca(fds, 500,logger)
    fds = np.array(fds, dtype=float)

    for i in range(birds):
        pos.append([])
        speed.append([])
        bestpos.append([])
        initpos.append([])
        tempfit.append([])

    for i in range(birds):  # initial all birds' pos,speed
        pos[i].append(random.uniform(10, 30))
        pos[i].append(random.uniform(0.5e-06, 1e-06))  # 1/num_features
        speed[i].append(float(0))
        speed[i].append(float(0))
        bestpos[i] = copy.deepcopy(pos[i])
        initpos[i] = copy.deepcopy(pos[i])

    logger.info("\n-------------------------Initial Globel Best Pos----------------------------------\n")
    best_predict, birdsbestpos = FindBirdsMostPos(logger,fds)  # initial birdsbestpos
    logger.info("\n-------------------------Done Globel Best Pos----------------------------------\n")

    for asd in range(maxgen):
        logger.info("\n>>>>>>>>The %d'd time to update parameters. Total %d times\n" % (asd + 1, maxgen))
        UpdateSpeed()
        best_predict, best_para = UpdatePos(logger,fds)

        allbestpos.append([best_para, best_predict])
        f = open('result/PSO_%s-%s-%s.txt' % (birds, maxgen, n), 'w')
        f.write(str(allbestpos))
        f.close()

    logger.info("After %d iterations\nthe best C is: %f\nthe best gamma is: %f" % (maxgen, best_para[0], best_para[1]))

