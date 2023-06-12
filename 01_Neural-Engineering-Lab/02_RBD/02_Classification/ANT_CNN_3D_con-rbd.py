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

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다. #
seed = 0
np.random.seed(seed)
start_time = timeit.default_timer()             # 시작 시간 체크

# load data #
data1 = mat73.loadmat('D:/ANT_3D_CNN/ANT_topo/control/ANT_con_topo_3d.mat')
data2 = mat73.loadmat('D:/ANT_3D_CNN/ANT_topo/rbd/ANT_rbd_topo_3d.mat')
#data1 = mat73.loadmat('D:/ANT_3D_CNN/ANT_topo/control/ANT_con_topo_3d_ERP_normalization.mat')
#data2 = mat73.loadmat('D:/ANT_3D_CNN/ANT_topo/rbd/ANT_rbd_topo_3d_ERP_normalization.mat')

# data 가공 #
con_1000 = data1['ANT_con_topo_3d']                                # con/rbd 각 1000 trials
rbd_1000 = data2['ANT_rbd_topo_3d']
#
rbd_z = np.random.choice(int(np.size(rbd_1000, axis=0)), int(np.size(con_1000, axis=0)), replace=False)
rbd_random = np.sort(rbd_z)
rbd_1000 = rbd_1000[rbd_random, :, :, :]
#
con_label = np.zeros(shape=(np.size(con_1000, axis=0), 1))
rbd_label = np.ones(shape=(np.size(rbd_1000, axis=0), 1))

gt_con = data1['ground_truth']
gt_rbd = data2['ground_truth']
del data1, data2

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
gt_rbd_nmci = np.sort(gt_rbd_nmci)


con_rbd = np.concatenate((con_1000, rbd_1000), axis=0)
con_rbd = np.nan_to_num(con_rbd)
con_rbd_label = np.concatenate((con_label, rbd_label), axis=0)
del con_1000, rbd_1000, con_label, rbd_label

# train/validation #
sf = np.random.choice(int(np.size(con_rbd, axis=0)), int(np.size(con_rbd, axis=0)), replace=False)
zz = np.reshape(con_rbd, (-1, 67, 67, 14, 1))
zzz = con_rbd_label
zz = zz[sf, :, :, :, :]
zzz = zzz[sf, :]
zzzz = np_utils.to_categorical(zzz)
del con_rbd, con_rbd_label

#random_index = np.random.choice(48160, 10000, replace=False)          # con or rbd 3000개 무작위로 뽑기
#random_index = np.sort(random_index)
#zz = zz[random_index, :, :, :, :]
#zzzz = zzzz[random_index, :]

#x_train, x_test, y_train, y_test = train_test_split(zz, zzzz, random_state=seed, shuffle=True, test_size=0.2)

# k-fold C.V. #
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
i = 1
accuracy = []

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

    history = model.fit(zz[train], zzzz[train], epochs=50, batch_size=128, verbose=2, callbacks=[early_stopping_callback])
    #history = model.fit(zz[train], zzzz[train], epochs=50, batch_size=128, verbose=2)

    k_accuracy = "%.8f" % (model.evaluate(zz[test], zzzz[test])[1])
    print("\n 현재 fold accuracy :", k_accuracy)
    accuracy.append(k_accuracy)

    #model.save('ANT_CNN_3D_con-rbd_%d.h5' % i)
    #model.save_weights('C:/Users/Nelab_001/PycharmProjects/pythonProject/ANT_CNN_3D_con-rbd_model_10-fold_weights/ANT_CNN_3D_con-rbd_%d.h5' % i)
    i += 1

print("\n %.8f fold accuracy : " % n_fold, accuracy)
np_acc = np.array(accuracy)
np_acc = np_acc.astype(np.double)
avg_acc = np.mean(np_acc)
std_acc = np.std(np_acc)
print(" %.8f fold 평균 정확도 : " % n_fold, avg_acc)

terminate_time = timeit.default_timer()         # 종료 시간 체크
print("%f초 걸렸습니다." % (terminate_time - start_time))     # 딥러닝 돌아가는데 걸린 시간


# 이미 만들어 놓은 모델을 불러와 weights 저장
fold_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in fold_num:
    model = load_model('C:/Users/Nelab_001/PycharmProjects/pythonProject/ANT_CNN_3D_con-rbd_model_10-fold/ANT_CNN_3D_con-rbd_%i.h5' % i)
    model.save_weights('C:/Users/Nelab_001/PycharmProjects/pythonProject/ANT_CNN_3D_con-rbd_model_10-fold_weights/ANT_CNN_3D_con-rbd_%i.h5' % i)
    del model
