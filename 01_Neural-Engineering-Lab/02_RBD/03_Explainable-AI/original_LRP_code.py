from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from keras.callbacks import EarlyStopping
import keras
from keras import layers
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
#start_time = timeit.default_timer()             # 시작 시간 체크

# load data #
data1 = mat73.loadmat('D:/ANT_3D_CNN/ANT_topo/control/ANT_con_topo_3d.mat')
data2 = mat73.loadmat('D:/ANT_3D_CNN/ANT_topo/rbd/ANT_rbd_topo_3d.mat')

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

# 3 class 만들기
# gt_con = data1['ground_truth']
# gt_rbd = data2['ground_truth']
#
# rbd_mci_index = np.array([2, 3, 4, 5, 11, 12, 15, 17, 19, 20, 21, 23, 24, 25, 26, 35, 38, 39, 41, 47, 49, 56, 59, 61])
# rbd_nmci_index = np.array([1, 6, 7, 8, 9, 10, 13, 14, 16, 18, 22, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 40, 42, 43, 44, 45, 46, 48, 50, 51, 52, 53, 54, 55, 57, 58, 60, 62])
# gt_rbd_mci = np.array([])
# gt_rbd_nmci = np.array([])
# for k in rbd_mci_index:
#     gt_rbd_mci_index = np.where(gt_rbd == k)
#     gt_rbd_mci = np.append(gt_rbd_mci, np.array(gt_rbd_mci_index))
# for l in rbd_nmci_index:
#     gt_rbd_nmci_index = np.where(gt_rbd == l)
#     gt_rbd_nmci = np.append(gt_rbd_nmci, np.array(gt_rbd_nmci_index))
# gt_rbd_mci = np.sort(gt_rbd_mci)
# gt_rbd_mci = np.array(gt_rbd_mci, dtype=np.int)
# gt_rbd_nmci = np.sort(gt_rbd_nmci)
# gt_rbd_nmci = np.array(gt_rbd_nmci, dtype=np.int)


# data 가공 #
# con_1000 = data1['ANT_con_topo_3d']                                # con/rbd 각 1000 trials
# rbd_1000 = data2['ANT_rbd_topo_3d']
# con_1000 = rbd_1000[gt_rbd_nmci, :, :, :]
# rbd_1000 = rbd_1000[gt_rbd_mci, :, :, :]
# con_label = np.zeros(shape=(np.size(con_1000, axis=0), 1))
# rbd_label = np.ones(shape=(np.size(rbd_1000, axis=0), 1))
# del data1, data2
#
# con_rbd = np.concatenate((con_1000, rbd_1000), axis=0)
# con_rbd = np.nan_to_num(con_rbd)
# con_rbd_label = np.concatenate((con_label, rbd_label), axis=0)
# del con_1000, rbd_1000, con_label, rbd_label

#random_index = np.random.choice(48160, 10000, replace=False)          # con or rbd 3000개 무작위로 뽑기
#random_index = np.sort(random_index)
#zz = zz[random_index, :, :, :, :]
#zzzz = zzzz[random_index, :]

x_train, x_test, y_train, y_test = train_test_split(zz, zzzz, random_state=seed, shuffle=True, test_size=0.1)
#x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=seed, shuffle=True, test_size=0.5)

model = Sequential()
model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), input_shape=(67, 67, 14, 1), activation='relu'))
model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), activation='relu'))
model.add(MaxPooling3D(pool_size=2))
model.add(Dropout(0.25))    # turn off of 25% nodes
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

#early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(x_train, y_train, epochs=20, batch_size=100, verbose=2)
#history = model.fit(x_train, y_train, epochs=20, batch_size=100, validation_data=(x_val, y_val), verbose=2)

k_accuracy = "%.4f" % (model.evaluate(x_test, y_test)[1])
print("\n 현재 fold accuracy :", k_accuracy)

#model.save('ANT_CNN_3D_topo3_rbd+nmci-rbd+mci.h5')

# LRP #
import innvestigate
import innvestigate.utils

model = load_model('C:/Users/Nelab_001/PycharmProjects/pythonProject/ANT_CNN_3D_con-rbd_model_10-fold/ANT_CNN_3D_con-rbd_1.h5')
#model = load_model('ANT_CNN_3D_con-rbd_1.h5')
y_predict = model.predict(x_test)
y_predict_con = np.reshape(np.where(y_predict[:,0]>0.5, 1, 0), (np.size(y_predict, axis=0), 1))     # CON 예측 결과
y_predict_rbd = np.reshape(np.where(y_predict[:,1]>0.5, 1, 0), (np.size(y_predict, axis=0), 1))     # RBD 예측 결과
y_predict_class = np.concatenate((y_predict_con, y_predict_rbd), axis=1)                            # 모델 예측 label

correct_index = np.array(np.where(y_test[:, 0] == y_predict_class[:, 0]))
correct_y_test = np.reshape(y_test[correct_index, :], (-1, 2))
correct_x_test = np.reshape(x_test[correct_index, :], (-1, 67, 67, 14, 1))
correct_index_con = np.array(np.where(correct_y_test[:, 0] == 1))
correct_index_rbd = np.array(np.where(correct_y_test[:, 1] == 1))
correct_x_test_con = np.reshape(correct_x_test[correct_index_con, :], (-1, 67, 67, 14, 1))                          # 정상인을 맞게 예측한 데이터(TP)
correct_x_test_rbd = np.reshape(correct_x_test[correct_index_rbd, :], (-1, 67, 67, 14, 1))                          # 환자를 맞게 예측한 데이터(TN)

incorrect_index = np.array(np.where(y_test[:, 0] != y_predict_class[:, 0]))
incorrect_y_test = np.squeeze(y_test[incorrect_index, :])
incorrect_x_test = np.reshape(x_test[incorrect_index, :, :, :], (-1, 67, 67, 14, 1))
incorrect_index_con = np.array(np.where(incorrect_y_test[:, 0] == 1))
incorrect_index_rbd = np.array(np.where(incorrect_y_test[:, 1] == 1))
incorrect_x_test_con = np.reshape(incorrect_x_test[incorrect_index_con, :, :, :], (-1, 67, 67, 14, 1))               # 정상인을 환자로 예측(FN)
incorrect_x_test_rbd = np.reshape(incorrect_x_test[incorrect_index_rbd, :, :, :], (-1, 67, 67, 14, 1))               # 환자를 정상인으로 예측(FP)

model_wo_softmax = innvestigate.utils.model_wo_softmax(model)                   # model without softmax output
analyzer = innvestigate.create_analyzer("deep_taylor", model_wo_softmax)              # XAI method

TN = np.zeros(np.shape(correct_x_test_rbd))
for m in range(np.size(correct_x_test_rbd, 0)):
    analysed_results_TN = analyzer.analyze(np.reshape(correct_x_test_rbd[m, :, :, :, :], (1, 67, 67, 14, 1)))
    TN[m, :, :, :, :] = analysed_results_TN
analysed_results_TN_mean = np.squeeze(np.mean(TN, axis=0))

TP = np.zeros(np.shape(correct_x_test_con))
for n in range(np.size(correct_x_test_con, 0)):
    analysed_results_TP = analyzer.analyze(np.reshape(correct_x_test_con[n, :, :, :, :], (1, 67, 67, 14, 1)))
    TP[n, :, :, :, :] = analysed_results_TP
analysed_results_TP_mean = np.squeeze(np.mean(TP, axis=0))

# save numpy_array to .mat file
np.save('C:/Users/Nelab_001/Documents/MATLAB/RBD_ANT/LRP/analysed_results_TN_mean4.npy', analysed_results_TP_mean)
np.save('C:/Users/Nelab_001/Documents/MATLAB/RBD_ANT/LRP/analysed_results_TN_mean5.npy', analysed_results_TN_mean)

#analysed_results_TP = analyzer.analyze(correct_x_test_con)                                    # XAI result
#analysed_results_TN = analyzer.analyze(correct_x_test_rbd)                                    # XAI result
#analysed_results_FN = analyzer.analyze(incorrect_x_test_con)                                    # XAI result
#analysed_results_FP = analyzer.analyze(incorrect_x_test_rbd)                                    # XAI result
#analysed_results_TP_mean = np.squeeze(np.mean(analysed_results_TP, axis=0))
#analysed_results_TN_mean = np.squeeze(np.mean(analysed_results_TN, axis=0))
#analysed_results_FN_mean = np.squeeze(np.mean(analysed_results_FN, axis=0))
#analysed_results_FP_mean = np.squeeze(np.mean(analysed_results_FP, axis=0))

plt.figure(0)
for i in range(np.size(analysed_results_TN_mean, 2)):
    a = analysed_results_TN_mean[:, :, i]
    b = np.reshape(a, (1, -1))
    c = np.sort(b)
    d = int(np.around(np.size(c)*0.05))  # 5%
    e = c[:, -d]

    f = np.where(a>=e, a, 0)
    g = np.where(a>=e, 1, 0)

    plt.subplot(2, 7, i+1)
    plt.imshow(f)
    plt.clim(0, 0.0008)
    #plt.colorbar()
plt.show()


plt.figure(1)
for i in range(np.size(analysed_results_TP_mean, 2)):
    a = analysed_results_TP_mean[:, :, i]
    b = np.reshape(a, (1, -1))
    c = np.sort(b)
    d = int(np.around(np.size(c)*0.05))  # 5%
    e = c[:, -d]

    f = np.where(a>=e, a, 0)
    g = np.where(a>=e, 1, 0)

    plt.subplot(2, 7, i+1)
    plt.imshow(f, cmap='jet')
    plt.clim(0, 0.0008)
    plt.colorbar()
plt.show()
