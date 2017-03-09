# -*- encoding: utf8 -*-
from numpy import *
import numpy as np

def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


def distCos(vecA, vecB):
    num = vecA * vecB.T
    denum = np.linalg.norm(vecA) * np.linalg.norm(vecB)
    if denum == 0:
        return 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn
    return sim


def randCent(dataSet, k):
    m, n = shape(dataSet)
    res = mat(zeros((k, n)))
    for i in range(n):
        minI = min(dataSet[:, i])
        rangeI = float(max(dataSet[:, i]) - minI)
        res[:, i] = minI + rangeI * np.random.rand(k, 1)
    return res


def kmeans(dataSet, k, distEclud=distCos, randCent=randCent):
    dataSet = np.mat(dataSet)
    m = np.shape(dataSet)[0]
    clust = np.mat(np.zeros((m, 2)))
    cent = randCent(dataSet, k)
    flag = True

    cnt = 0
    while flag:
        flag = False
        cnt+=1
        for i in range(m):
            minDist = np.inf; minIdx = -1
            for j in range(k):
                dist = distEclud(dataSet[i, :], cent[j, :])
                if minDist > dist:
                    minDist = dist; minIdx = j
            if clust[i, 0] != minIdx:
                flag = True
            clust[i, :] = minIdx, minDist**2

        for i in range(k):
            ptsInClust = dataSet[np.nonzero(clust[:, 0].A == i)[0]]
            if len(ptsInClust) == 0 : continue
            cent[i, :] = np.mean(ptsInClust, axis=0)
        if cnt>50:break
    return clust, cent


def biKmeans(dataSet, k, distMeans=distCos):
    m = shape(dataSet)[0]
    clust = mat(zeros((m, 2)))
    cent = mean(dataSet, axis=0).tolist()[0]
    centList = [cent]  # create a list with one centroid
    for i in range(m):
        clust[i, 1] = distMeans(np.mat(cent), dataSet[i, :])
    while len(centList) < k:
        minSSE = np.inf
        for i in range(len(centList)):
            ptsInCurClust = dataSet[np.nonzero(clust[:, 0].A == i)[0], :]
            subClust, subCent = kmeans(ptsInCurClust, 2, distMeans)
            sseSubClust = np.sum(subClust[:, 1])
            sseNotSubClust = np.sum(clust[np.nonzero(clust[:, 0].A != i)[0], 1])
            if sseSubClust + sseNotSubClust < minSSE:
                minIdx = i
                minSubCent = subCent
                minSubClust = subClust.copy()
                minSSE = sseSubClust + sseNotSubClust
        minSubClust[np.nonzero(minSubClust[:, 0] == 1)[0], 0] = len(centList)
        minSubClust[np.nonzero(minSubClust[:, 0] == 0)[0], 0] = minIdx
        print 'the bestCentToSplit is: ', minIdx
        print 'the len of bestClustAss is: ', len(minSubClust)
        centList[minIdx] = minSubCent[0, :].tolist()[0]
        centList.append(minSubCent[1, :].tolist()[0])
        clust[np.nonzero(clust[:, 0].A == minIdx)[0], :] = minSubClust
    return clust, mat(centList)
