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
result_file_name = "PCA_testing_120"
errors_pca = pd.read_csv(results_folder_path + result_file_name + "_errors.csv", header=None).values.flatten()
time_pca = pd.read_csv(results_folder_path + result_file_name + "_times.csv", header=None).values.flatten()
variance_pca = pd.read_csv(results_folder_path + result_file_name + "_variance.csv", header=None).values.flatten()


result_file_name = "KPCA_testing_120"
errors_kpca = pd.read_csv(results_folder_path + result_file_name + "_errors.csv", header=None).values.flatten()
time_kpca = pd.read_csv(results_folder_path + result_file_name + "_times.csv", header=None).values.flatten()
variance_kpca = pd.read_csv(results_folder_path + result_file_name + "_variance.csv", header=None).values.flatten()

fig, axes = plt.subplots(1,1)
axes.semilogy(range(nmax),(1-errors_pca)*100, label="PCA")
axes.semilogy(range(nmax),(1-errors_kpca)*100, label="KPCA")
axes.grid(which='Both')
plt.xlabel('No. autocaras', fontsize=20)
plt.ylabel('Error', fontsize=20)
plt.tight_layout()
plt.legend(loc=1, prop={'size': 17})

fig, axes = plt.subplots(1,1)
axes.plot(range(nmax),time_pca, label="PCA")
axes.plot(range(nmax),time_kpca, label="KPCA")
plt.xlabel('No. autocaras', fontsize=20)
plt.ylabel('Tiempo', fontsize=20)
axes.grid(which='Both')
plt.legend(loc=4, prop={'size': 17})

fig, axes = plt.subplots(1,1)
axes.plot(range(nmax),variance_pca, label="PCA")
axes.plot(range(nmax),variance_kpca, label="KPCA")
plt.xlabel('No. autocaras', fontsize=20)
plt.ylabel('Varianza', fontsize=20)
axes.grid(which='Both')
plt.legend(loc=4, prop={'size': 17})


fig, axes = plt.subplots(1,1)
axes.semilogy(variance_pca,(1-errors_pca)*100, label="PCA")
axes.semilogy(variance_kpca,(1-errors_kpca)*100, label="KPCA")
axes.set_xlabel('Varianza', fontsize=20)
plt.ylabel('Error', fontsize=20)
axes.grid(which='Both')
plt.legend(loc=1, prop={'size': 17})


results_folder_path = "../results/"
result_file_name = "PCA_testing_people_80"

errors_people_pca = pd.read_csv(results_folder_path + result_file_name + "_errors.csv", header=None).values.flatten()
time_people_pca = pd.read_csv(results_folder_path + result_file_name + "_times.csv", header=None).values.flatten()

result_file_name = "KPCA_testing_people_80_v2"

errors_people_kpca = pd.read_csv(results_folder_path + result_file_name + "_errors.csv", header=None).values.flatten()
time_people_kpca = pd.read_csv(results_folder_path + result_file_name + "_times.csv", header=None).values.flatten()


fig, axes = plt.subplots(1,1)
axes.semilogy(range(2, len(errors_people_pca[2:])+2),(1.01-errors_people_pca[2:])*100, label="PCA")
axes.semilogy(range(2, len(errors_people_kpca[2:])+2),(1.01-errors_people_kpca[2:])*100, label="KPCA")
axes.grid(which='Both')
plt.xlabel('No. personas', fontsize=20)
plt.ylabel('Error', fontsize=20)
plt.tight_layout()
plt.legend(loc=1, prop={'size': 17})


fig, axes = plt.subplots(1,1)
axes.plot(range(2, len(errors_people_pca[2:])+2),time_people_pca[2:], label="PCA")
axes.plot(range(2, len(errors_people_kpca[2:])+2),time_people_kpca[2:], label="KPCA")
plt.xlabel('No. personas', fontsize=20)
plt.ylabel('Tiempo', fontsize=20)
axes.grid(which='Both')
plt.legend(loc=4, prop={'size': 17})

plt.show()