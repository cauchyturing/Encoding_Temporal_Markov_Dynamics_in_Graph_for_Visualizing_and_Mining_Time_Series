# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 12:36:45 2014

@author: Stephen Wang
"""
import serie2QMlib as lib
import os
import bop
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def QMeq(series, Q):
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.codes))
    MSM = np.zeros([Q, Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0, len(label) - 1):
        MSM[label[i]][label[i + 1]] += 1
    for i in range(Q):
        if sum(MSM[i][:]) == 0:
            continue
        # MSM[i][:] = np.exp(MSM[i][:])-1
        MSM[i][:] = MSM[i][:] / sum(MSM[i][:])
    return np.array(MSM), label


def movingaverage(values, window, sma_bool=True):
    if sma_bool:
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
    else:
        sma = values
    return sma


# %%
def PAA_field(data, s, blur_bool=True):
    if blur_bool:
        batch = len(data) / s
        patch = []
        for p in range(s):
            for q in range(s):
                patch.append(np.mean(data[p * batch:(p + 1) * batch, q * batch:(q + 1) * batch]))
        patch = np.array(patch)
        patch_img = np.array(patch).reshape(s, s)
        patch_img = patch_img.T / patch_img.sum(axis=1)
    else:
        patch_img = data

    return patch_img.T


# %%
alphabetSize = 50
quantile = alphabetSize

fileName = 'Gun_Point_TEST'
sma_bool = False  # Moving average?
blur_bool = False  # Reduce MTF size?
mv_step = 0.01  # Moving average step size
s = 64  # Size of blurred MTF

# No use
skipSize = 1
numSymbols = -1  # -1
windowWidth = 1  # -1

dirM = fileName + '_' + str(alphabetSize) + '_SAX_M_sma_' + str(sma_bool) + '_' + str(mv_step) + '/'
dirMTF = fileName + '_' + str(alphabetSize) + '_SAX_MTF_sma_' + str(sma_bool) + '_' + str(mv_step) + '_blur_' + str(
    s) + '/'

if not os.path.isdir(dirM):
    os.makedirs(dirM)
if not os.path.isdir(dirMTF):
    os.makedirs(dirMTF)
outputM = dirM + fileName + '-quantile-' + str(quantile)
outputMTF = dirMTF + fileName + '-quantile-' + str(quantile)

# alphabetSize = int(sys.argv[3])

print('skipSize:', skipSize)
SAXqt = []
SAXQM = []
SAXlabel = []
SAXMTF = []
TS = []
raw = open(fileName).readlines()
for i in range(len(raw)):
    rawdata = raw[i].strip().split(',')[1:]
    rawdata = [float(eachdata) for eachdata in rawdata]
    label = float(raw[i].strip().split(',')[0])
    TS.append(rawdata)
    rawdata = movingaverage(rawdata, int(mv_step * len(rawdata)), sma_bool=sma_bool)
    if numSymbols == -1:
        numSymbols = len(rawdata)
    if 0 < windowWidth <= 1:
        windowWidth = int(len(rawdata) * windowWidth)

    data = bop.standardize(rawdata)
    data = bop.sax_words(data, windowWidth, skipSize, numSymbols, None, alphabetSize, True)
    peSeries = data[0].split('_')
    peSeries.insert(0, label)
    peSeries = list(map(float, peSeries))
    SAXqt.append(peSeries)
    qv = lib.QVeq(peSeries[1:], quantile)
    # Generate Markov Matrix
    mat, matindex = QMeq(peSeries[1:], quantile)
    SAXQM.append(mat)

    column = []
    for p in range(len(rawdata)):
        for q in range(len(rawdata)):
            column.append(mat[matindex[p]][matindex[(q)]])
    column = np.array(column)
    columnmatrix = column.reshape(len(rawdata), len(rawdata))
    MTF_PAA = PAA_field(columnmatrix, s, blur_bool=blur_bool)
    SAXMTF.append(MTF_PAA)
    lib.writeQM(mat, outputM + '-label' + str(label) + '-' + str(i) + '.csv')
    lib.writeQM(MTF_PAA, outputMTF + '-label' + str(label) + '-' + str(i) + '.csv')

fileName = 'Gun_Point_TEST'
raw = open(fileName).readlines()
TS = []
for each in raw:
    rawdata = each.strip().split(',')
    rawdata = [float(eachdata) for eachdata in rawdata]
    TS.append(rawdata)

plt.figure()
plt.plot(np.arange(len(TS[0])), TS[0], linewidth=5.0);
plt.xticks([], [])
plt.yticks([], [])