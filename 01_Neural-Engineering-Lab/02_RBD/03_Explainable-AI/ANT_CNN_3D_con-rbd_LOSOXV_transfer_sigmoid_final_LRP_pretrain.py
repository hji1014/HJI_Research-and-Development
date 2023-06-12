from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import keras
from keras import layers
import pandas as pd
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
import innvestigate
import innvestigate.utils
from sklearn.metrics import confusion_matrix


# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다. #
seed = 0
np.random.seed(seed)
start_time = timeit.default_timer()             # 시작 시간 체크

for sub in range(62):
    print('---------------- 현재 처리 중인 피험자 번호 :', (sub + 1), '----------------')

    transfer_model = Sequential()
    transfer_model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), input_shape=(67, 67, 14, 1), activation='relu'))
    transfer_model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), activation='relu'))
    transfer_model.add(MaxPooling3D(pool_size=2))
    transfer_model.add(Dropout(0.25))  # turn off of 25% nodes
    transfer_model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), activation='relu'))
    transfer_model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), activation='relu'))
    transfer_model.add(MaxPooling3D(pool_size=2))
    transfer_model.add(Dropout(0.25))
    transfer_model.add(Flatten())
    transfer_model.add(Dense(256, activation='relu'))
    transfer_model.add(Dropout(0.5))
    transfer_model.add(Dense(128, activation='relu'))
    transfer_model.add(Dropout(0.5))
    transfer_model.add(Dense(1, activation='sigmoid'))
    transfer_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    transfer_model.summary()
    transfer_model.load_weights('D:/ANT_3D_CNN_transfer2/premodel_weights/premodel_weights_sub_%d.h5' % (sub + 1))

    x_test = np.load('D:/ANT_3D_CNN_transfer2/premodel_testset/premodel_x_sub_%d.npy' % (sub + 1))
    y_test = np.load('D:/ANT_3D_CNN_transfer2/premodel_testset/premodel_y_sub_%d.npy' % (sub + 1))

    # confusion matrix
    k_accuracy = "%.4f" % (transfer_model.evaluate(x_test, y_test)[1])
    print("\n 현재 fold accuracy :", k_accuracy)                              # test 정확도
    y_predict = transfer_model.predict(x_test)
    y_predict_accurate = np.where(y_predict>=0.5, 1, 0)
    cf = confusion_matrix(y_test, y_predict_accurate)
    print('*********** confusion matrix 는 ***********\n', cf)

    TN_num = cf[1, 1]           # RBD를 RBD로 예측(TP)
    FN_num = cf[1, 0]           # RBD를 NC로 예측(FN)

    y_predict_accurate_onehot = np_utils.to_categorical(y_predict_accurate)
    y_test_onehot = np_utils.to_categorical(y_test)

    correct_index = np.array(np.where(y_test_onehot[:, 0] == y_predict_accurate_onehot[:, 0]))
    correct_y_test = np.reshape(y_test_onehot[correct_index, :], (-1, 2))
    correct_x_test = np.reshape(x_test[correct_index, :], (-1, 67, 67, 14, 1))
    correct_index_con = np.array(np.where(correct_y_test[:, 0] == 1))
    correct_index_rbd = np.array(np.where(correct_y_test[:, 1] == 1))
    correct_x_test_con = np.reshape(correct_x_test[correct_index_con, :], (-1, 67, 67, 14, 1))  # 정상인을 맞게 예측한 데이터(TN)
    correct_x_test_rbd = np.reshape(correct_x_test[correct_index_rbd, :], (-1, 67, 67, 14, 1))  # 환자를 맞게 예측한 데이터(TP)

    incorrect_index = np.array(np.where(y_test_onehot[:, 0] != y_predict_accurate_onehot[:, 0]))
    incorrect_y_test = np.reshape(y_test_onehot[incorrect_index, :], (-1, 2))
    incorrect_x_test = np.reshape(x_test[incorrect_index, :, :, :], (-1, 67, 67, 14, 1))
    incorrect_index_con = np.array(np.where(incorrect_y_test[:, 0] == 1))
    incorrect_index_rbd = np.array(np.where(incorrect_y_test[:, 1] == 1))
    incorrect_x_test_con = np.reshape(incorrect_x_test[incorrect_index_con, :, :, :], (-1, 67, 67, 14, 1))  # 정상인을 환자로 예측(FP)
    incorrect_x_test_rbd = np.reshape(incorrect_x_test[incorrect_index_rbd, :, :, :], (-1, 67, 67, 14, 1))  # 환자를 정상인으로 예측(FN)

    analyzer = innvestigate.create_analyzer("deep_taylor", transfer_model)  # XAI method

    TP = np.zeros(np.shape(correct_x_test_rbd))
    print('*********** TN에 대한 LRP 계산 중... ***********')
    for m in range(np.size(correct_x_test_rbd, axis=0)):
        analysed_results_TP = analyzer.analyze(np.reshape(correct_x_test_rbd[m, :, :, :, :], (1, 67, 67, 14, 1)))
        TP[m, :, :, :, :] = analysed_results_TP

    # save numpy_array to .mat file
    np.save('D:/ANT_3D_CNN_transfer2/LRP/LRP_TP_sub_%d.npy' % (sub + 1), TP)

    #if np.size(incorrect_x_test_rbd, axis=0) == 0:
    #    print('*********** FN인 경우가 없습니다... ***********')
    #else:
    #    FN = np.zeros(np.shape(incorrect_x_test_rbd))
    #    print('*********** FN에 대한 LRP 계산 중... ***********')
    #    for n in range(np.size(incorrect_x_test_rbd, axis=0)):
    #        analysed_results_FN = analyzer.analyze(np.reshape(incorrect_x_test_rbd[n, :, :, :, :], (1, 67, 67, 14, 1)))
    #        FN[n, :, :, :, :] = analysed_results_FN
    #
    #    # save numpy_array to .mat file
    #    np.save('D:/ANT_3D_CNN_transfer2/LRP/LRP_FN_sub_%d.npy' % (sub + 1), FN)

terminate_time = timeit.default_timer()         # 종료 시간 체크
print("%f초 걸렸습니다." % (terminate_time - start_time))     # 딥러닝 돌아가는데 걸린 시간

pre_acc = np.load('D:/ANT_3D_CNN_transfer/premodel_accuracy/all_PT_acc.npy')
tl_acc = np.load('D:/ANT_3D_CNN_transfer/transfer_model_accuracy/all_TL_acc.npy')

pre_acc_num = pre_acc.astype(np.float) * 100
tl_acc_num = tl_acc.astype(np.float) * 100

a = np.mean(pre_acc_num)
b = np.mean(tl_acc_num)
aa = np.std(pre_acc_num)
bb = np.std(tl_acc_num)
