
from os import listdir
from os.path import join, isdir

import matplotlib
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sys import stdin
import cv2
import src.face_detector as fd


from utils import *

import argparse

from src.svd import svd, _eig
from src.utils import openImages

parser = argparse.ArgumentParser(description='Facial recognition system.')
parser.add_argument("--kernel", "-k", help="Uses KPCA", action="store_true",
                    default=False)
parser.add_argument("--faces_directory", help="Path to the directory with the faces.", action="store",
                    default='./../att_faces/')
parser.add_argument("--face_test_directory", help="Path to the directory with the faces to test.", action="store",
                    default='./../att_faces/')
parser.add_argument("--eigenfaces", help="How many eigenfaces are used.", action="store", default=50)
parser.add_argument("--training", help="How many photos used for training out of 10.", action="store",
                    choices=[1,2,3,4,5,6,7,8,9,10], type=int, default=6)
args = parser.parse_args()

mypath = args.faces_directory

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = 42
trnperper   = args.training
tstperper   = 10 - args.training
trnno = personno * trnperper
tstno = personno * tstperper

clf = svm.LinearSVC()

# TRAINING

images_training, person_training = openImages(path=mypath, personno=personno, trnperper=trnperper, areasize=areasize)
if args.kernel:
    images_training *= 255.0
    images_training -= 127.5
    images_training /= 127.5

    # KERNEL: polinomial de grado degree
    degree = 2
    K = (np.dot(images_training, images_training.T) / trnno + 1) ** degree
    # esta transformación es equivalente a centrar las imágenes originales...
    unoM = np.ones([trnno, trnno]) / trnno
    K = K - np.dot(unoM, K) - np.dot(K, unoM) + np.dot(unoM, np.dot(K, unoM))
    # Autovalores y autovectores
    w, alpha = _eig(K)
    lambdas = w
    for col in range(alpha.shape[1]):
        alpha[:, col] = alpha[:, col] / np.sqrt(lambdas[col])

    # pre-proyección
    improypre = np.dot(K.T, alpha)
    proy_training = improypre[:, 0:args.eigenfaces]

else:

    # CARA MEDIA
    meanimage = np.mean(images_training, 0)
    images_training = [images_training[k, :] - meanimage for k in range(images_training.shape[0])]
    # PCA
    images_training = np.asarray(images_training)
    V = svd(images_training)

    B = V[0:args.eigenfaces, :]
    proy_training = np.dot(images_training, B.T)

clf.fit(proy_training, person_training.ravel())


# TEST PICTURES
test_path = args.face_test_directory

video_capture = cv2.VideoCapture(0)
while(True):
    ret, frame = video_capture.read()

    if not ret:
        # print("The video capture is not working.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cascPath = "./haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(92, 112)
    )

    #Draw a rectangle around the faces
    for f in faces:
        x, y, w, h = fd.resizeFace(f)
        if x < 0 or y < 0 or x + w > 1280 or y + h > 1280:
            continue
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        a = cv2.resize(gray, (92,112))
        a = np.array(a).ravel()
        a = np.float64(a)

        if args.kernel:
            unoML = np.ones([1, trnno]) / trnno
            Ktest = (np.dot(a, images_training.T) / trnno + 1) ** degree
            Ktest = Ktest - np.dot(unoML, K) - np.dot(Ktest, unoM) + np.dot(unoML, np.dot(K, unoM))
            imtstproypre = np.dot(Ktest, alpha)
            proy_test = imtstproypre[:, 0:args.eigenfaces]
        else:
            a -= meanimage
            proy_test = np.dot(a, B.T)

        # cv2.putText(frame, "It is subject " + clf.predict(proy_test)[0].astype(int).astype(str) + "!",
        #             (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        print("It is subject " + clf.predict(proy_test)[0].astype(int).astype(str) + "!")

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # if cv2.waitKey(1) & 0xFF == ord(' ') and frame is not None and len(faces) > 0:
    #     newImg = fd.cropImage(frame, fd.resizeFace(faces[0]))
    #     newImg = fd.resizeImg(newImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()