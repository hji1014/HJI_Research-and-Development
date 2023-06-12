from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from tensorflow.keras.callbacks import EarlyStopping
import keras
from keras import layers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import h5py
import random
import mat73
import timeit
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
from keras import optimizers

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다. #

seed = 0
np.random.seed(seed)
start_time = timeit.default_timer()             # 시작 시간 체크

# load data #
data1 = mat73.loadmat('D:/ANT_3D_CNN/ANT_topo/control/ANT_con_topo_3d.mat')
data2 = mat73.loadmat('D:/ANT_3D_CNN/ANT_topo/rbd/ANT_rbd_topo_3d.mat')

gt_con = data1['ground_truth']
gt_rbd = data2['ground_truth']

rbd_mci_index = np.array([2, 3, 4, 5, 11, 12, 15, 17, 19, 20, 21, 23, 24, 25, 26, 35, 38, 39, 41, 47, 49, 56, 59, 61])
rbd_nmci_index = np.array([1, 6, 7, 8, 9, 10, 13, 14, 16, 18, 22, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 40, 42, 43, 44, 45, 46, 48, 50, 51, 52, 53, 54, 55, 57, 58, 60, 62])
gt_rbd_mci = np.array([])
gt_rbd_nmci = np.array([])
for k in rbd_mci_index:
    gt_rbd_mci_index = np.where(gt_rbd == k)
    gt_rbd_mci = np.append(gt_rbd_mci, np.array(gt_rbd_mci_index))
for l in rbd_nmci_index:
    gt_rbd_nmci_index = np.where(gt_rbd == l)
    gt_rbd_nmci = np.append(gt_rbd_nmci, np.array(gt_rbd_nmci_index))
gt_rbd_mci = np.sort(gt_rbd_mci)
gt_rbd_mci = np.array(gt_rbd_mci, dtype=np.int)
gt_rbd_nmci = np.sort(gt_rbd_nmci)
gt_rbd_nmci = np.array(gt_rbd_nmci, dtype=np.int)

# data 가공 #
con_1000 = data1['ANT_con_topo_3d']                                # con/rbd 각 1000 trials
rbd_1000 = data2['ANT_rbd_topo_3d']
rbd_1000 = rbd_1000[gt_rbd_nmci, :, :, :]
rbd_balance_idx = np.sort(np.random.choice(np.size(rbd_1000, axis=0), np.size(con_1000, axis=0), replace=False))
rbd_1000 = rbd_1000[rbd_balance_idx, :, :, :]
con_label = np.zeros(shape=(np.size(con_1000, axis=0), 1))
rbd_label = np.ones(shape=(np.size(rbd_1000, axis=0), 1))

del data1, data2

con_rbd = np.concatenate((con_1000, rbd_1000), axis=0)
con_rbd = np.nan_to_num(con_rbd)
con_rbd_label = np.concatenate((con_label, rbd_label), axis=0)
del con_1000, rbd_1000, con_label, rbd_label

# train/validation #

zz = np.reshape(con_rbd, (-1, 67, 67, 14, 1))
zzz = con_rbd_label
zzzz = np_utils.to_categorical(con_rbd_label)
del con_rbd, con_rbd_label

shuffle_idx = np.random.choice(np.size(zz, axis=0), np.size(zz, axis=0), replace=False)
zz = zz[shuffle_idx, :, :, :, :]
zzz = zzz[shuffle_idx, :]
zzzz = zzzz[shuffle_idx, :]

# k-fold C.V. #
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
i = 1

train_accuracy = []
test_accuracy = []

for train, test in skf.split(zz, zzz):

    model = Sequential()
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), input_shape=(67, 67, 14, 1), activation='relu'))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), activation='relu'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(0.25))  # turn off of 25% nodes
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), activation='relu'))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), activation='relu'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    early_stopping_callback = EarlyStopping(monitor='loss', patience=10)
    tensorboard = TensorBoard(log_dir="D:/ANT_3D_CNN_con-rbd+nmci/logs/{}".format(time()))

    history = model.fit(zz[train], zzzz[train], epochs=100, batch_size=128, verbose=2, callbacks=[early_stopping_callback, tensorboard])

    k_train_accuracy = "%.4f" % (model.evaluate(zz[train], zzzz[train])[1])
    k_test_accuracy = "%.4f" % (model.evaluate(zz[test], zzzz[test])[1])
    print("\n 현재 fold train accuracy :", k_train_accuracy)
    print("\n 현재 fold test accuracy :", k_test_accuracy)
    train_accuracy.append(k_train_accuracy)
    test_accuracy.append(k_test_accuracy)

    model.save_weights('D:/ANT_3D_CNN_con-rbd+nmci/weights/model_weights_fold_%d.h5' % i)
    np.save('D:/ANT_3D_CNN_con-rbd+nmci/testset/x_fold_%d.npy' % i, zz[test])
    np.save('D:/ANT_3D_CNN_con-rbd+nmci/testset/y_fold_%d.npy' % i, zzzz[test])

    i += 1

print("\n %.f fold train accuracy : " % n_fold, train_accuracy)
print("\n %.f fold test accuracy : " % n_fold, test_accuracy)
np_acc_train = np.array(train_accuracy)
np_acc_train = np_acc_train.astype(np.double)
avg_acc_train = np.mean(np_acc_train)
std_acc_train = np.std(np_acc_train)
np_acc_test = np.array(test_accuracy)
np_acc_test = np_acc_test.astype(np.double)
avg_acc_test = np.mean(np_acc_test)
std_acc_test = np.std(np_acc_test)
print(" %.f fold 평균 train 정확도 : " % n_fold, avg_acc_train)
print(" %.f fold 표준편차 train 정확도 : " % n_fold, std_acc_train)
print(" %.f fold 평균 test 정확도 : " % n_fold, avg_acc_test)
print(" %.f fold 표준편차 test 정확도 : " % n_fold, std_acc_test)

terminate_time = timeit.default_timer()         # 종료 시간 체크
print("%f초 걸렸습니다." % (terminate_time - start_time))     # 딥러닝 돌아가는데 걸린 시간
