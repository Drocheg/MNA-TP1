# -*- coding: utf-8 -*-
from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from svd import *
from svd import _eig

from svd import _eig
from src.svd import *
import pandas as pd
import time


mypath      = './../att_faces/'
onlydirs    = [f for f in listdir(mypath) if isdir(join(mypath, f))]

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = 40
trnperper   = 6
tstperper   = 4
trnno       = personno*trnperper
tstno       = personno*tstperper
accs = np.zeros([11, 1])
times = np.zeros([11, 1])

for degree in range(2, 11):

    #TRAINING SET
    images = np.zeros([trnno,areasize])
    person = np.zeros([trnno,1])
    imno = 0
    per  = 0
    for dire in onlydirs:
        for k in range(1,trnperper+1):
            a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')
            images[imno,:] = (np.reshape(a,[1,areasize])-127.5)/127.5
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
            a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')
            imagetst[imno,:]  = (np.reshape(a,[1,areasize])-127.5)/127.5
            persontst[imno,0] = per
            imno += 1
        per += 1
        if per >= personno:
            break

    start_time = time.time()

    #KERNEL: polinomial de grado degree
    K = (np.dot(images,images.T)/trnno+1)**degree
    #K = (K + K.T)/2.0

    #esta transformación es equivalente a centrar las imágenes originales...
    unoM = np.ones([trnno,trnno])/trnno
    K = K - np.dot(unoM,K) - np.dot(K,unoM) + np.dot(unoM,np.dot(K,unoM))


    #Autovalores y autovectores
    w,alpha = _eig(K)
    lambdas = w/trnno
    lambdas = w

    #Los autovalores vienen en orden descendente. Lo cambio
    #lambdas = np.flipud(lambdas)
    #alpha   = np.fliplr(alpha)

    for col in range(alpha.shape[1]):
        alpha[:,col] = alpha[:,col]/np.sqrt(lambdas[col])

    #pre-proyección
    improypre   = np.dot(K.T,alpha)
    unoML       = np.ones([tstno,trnno])/trnno
    Ktest       = (np.dot(imagetst,images.T)/trnno+1)**degree
    Ktest       = Ktest - np.dot(unoML,K) - np.dot(Ktest,unoM) + np.dot(unoML,np.dot(K,unoM))
    imtstproypre= np.dot(Ktest,alpha)

    #from sklearn.decomposition import KernelPCA

    #kpca = KernelPCA(n_components = None, kernel='poly', degree=2, gamma = 1, coef0 = 0)
    #kpca =  (n_components = None, kernel='poly', degree=2)
    #kpca.fit(images)

    #improypre = kpca.transform(images)
    #imtstproypre = kpca.transform(imagetst)


    #nmax = alpha.shape[1]
    neigen = 80

    improy      = improypre[:,0:neigen]
    imtstproy   = imtstproypre[:,0:neigen]

    clf = svm.LinearSVC()
    clf.fit(improy,person.ravel())
    times[degree] = time.time() - start_time
    accs[degree] = clf.score(imtstproy,persontst.ravel())
    print('Precisión con {0} autocaras: {1} %\n'.format(neigen,accs[degree]*100))

results_folder_path = "../results/"
result_file_name = "KPCA_testing_60_degree"

pd.DataFrame(accs).to_csv(results_folder_path + result_file_name + "_errors.csv", header=None, index=None)
pd.DataFrame(times).to_csv(results_folder_path + result_file_name + "_times.csv", header=None, index=None)
