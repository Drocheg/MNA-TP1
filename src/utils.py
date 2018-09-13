
from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from svd import *


def openImages(path, personno, trnperper, areasize):
    onlydirs = [f for f in listdir(path) if isdir(join(path, f))]
    images_no = personno * trnperper
    images = np.zeros([images_no, areasize])
    person = np.zeros([images_no, 1])
    imno = 0
    per = 1
    names = {}
    for dire in onlydirs:
        for k in range(1, trnperper + 1):
            a = im.imread(path + dire + '/{}'.format(k) + '.pgm') / 255.0
            images[imno, :] = np.reshape(a, [1, areasize])
            person[imno, 0] = per
            imno += 1
        names[per] = dire
        per += 1
        if per > personno:
            break
    return images, person, names

