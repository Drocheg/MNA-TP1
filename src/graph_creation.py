from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sys import stdin
from  sklearn.ensemble import GradientBoostingClassifier
from utils import *
import argparse
from src.svd import svd, _eig
from src.utils import openImages
import pandas as pd


nmax = 120 # CHANGE THIS TO SIZE OF EIGENVECTORS
results_folder_path = "../results/"
result_file_name = "KPCA_testing_120"
errors = pd.read_csv(results_folder_path + result_file_name + "_errors.csv", header=None).values.flatten()
time = pd.read_csv(results_folder_path + result_file_name + "_times.csv", header=None).values.flatten()
variance = pd.read_csv(results_folder_path + result_file_name + "_variance.csv", header=None).values.flatten()


fig, axes = plt.subplots(1,1)
axes.semilogy(range(nmax),(1-errors)*100)
axes.set_xlabel('No. autocaras')
axes.grid(which='Both')
fig.suptitle('Error')

fig, axes = plt.subplots(1,1)
axes.plot(range(nmax),time)
axes.set_xlabel('No. autocaras')
axes.grid(which='Both')
fig.suptitle('Time')
plt.show()

fig, axes = plt.subplots(1,1)
axes.plot(range(nmax),variance)
axes.set_xlabel('No. autocaras')
axes.grid(which='Both')
fig.suptitle('Variance')
plt.show()


fig, axes = plt.subplots(1,1)
axes.semilogy(variance,(1-errors)*100)
axes.set_xlabel('Varianza')
axes.grid(which='Both')
fig.suptitle('Variance-Error')
plt.show()

