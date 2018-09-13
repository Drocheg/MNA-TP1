# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:32:14 2017

@author: pfierens
"""
from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from src.svd import *
import pandas as pd
import time

mypath      = './../att_faces/'
onlydirs    = [f for f in listdir(mypath) if isdir(join(mypath, f))]

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

variance = 0.8

#number of figures
accs = np.zeros([41,1])
times = np.zeros([41, 1])
for people_num in range(2,41):
    personno    = people_num
    trnperper   = 6
    tstperper   = 4
    trnno       = personno*trnperper
    tstno       = personno*tstperper

    #TRAINING SET
    images = np.zeros([trnno,areasize])
    person = np.zeros([trnno,1])
    imno = 0
    per  = 0
    for dire in onlydirs:
        for k in range(1,trnperper+1):
            a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')/255.0
            images[imno,:] = np.reshape(a,[1,areasize])
            person[imno,0] = per
            imno += 1
        per += 1
        if per >= personno:
            break

    #TEST SET
    imagetst  = np.zeros([tstno,areasize])
    persontst = np.zeros([tstno,1])
    imno = 0
    per  = 0
    for dire in onlydirs:
        for k in range(trnperper,10):
            a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')/255.0
            imagetst[imno,:]  = np.reshape(a,[1,areasize])
            persontst[imno,0] = per
            imno += 1
        per += 1
        if per >= personno:
           break

    #CARA MEDIA
    meanimage = np.mean(images,0)

    #resto la media
    images  = [images[k,:]-meanimage for k in range(images.shape[0])]
    imagetst= [imagetst[k,:]-meanimage for k in range(imagetst.shape[0])]

    #PCA
    images = np.asarray(images)
    eigen_values, V = svd(images)

    #nmax = num_eigenvectors
    nmax = 120

   # neigen = eig_with_variance_explained(eigen_values, V, variance)
    neigen = 50
    print(variance_explained_from_eigen_values(eigen_values,neigen))

    start_time = time.time()
    B = V[0:neigen,:]
    improy      = np.dot(images, B.T)
    imtstproy   = np.dot(imagetst, B.T)
    clf = svm.LinearSVC()
    clf.fit(improy,person.ravel())
    end_time = time.time()
    times[people_num] = end_time - start_time
    accs[people_num] = clf.score(imtstproy,persontst.ravel())
    print('Precisión con {0} personas: {1} %\n'.format(people_num,accs[people_num]*100))

results_folder_path = "../results/"
result_file_name = "PCA_testing_people_50_caras"

pd.DataFrame(accs).to_csv(results_folder_path + result_file_name + "_errors.csv", header=None, index=None)
pd.DataFrame(times).to_csv(results_folder_path + result_file_name + "_times.csv", header=None, index=None)
