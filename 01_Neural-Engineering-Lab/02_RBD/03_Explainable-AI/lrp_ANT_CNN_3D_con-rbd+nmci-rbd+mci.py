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

for fold in range(5):
    print('---------------- 현재 처리 중인 fold 번호 :', (fold + 1), '----------------')

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
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.load_weights('D:/ANT_3D_CNN_con-rbd+nmci-rbd+mci/weights/model_weights_fold_%d.h5' % (fold + 1))

    x_test = np.load('D:/ANT_3D_CNN_con-rbd+nmci-rbd+mci/testset/x_fold_%d.npy' % (fold + 1))
    y_test = np.load('D:/ANT_3D_CNN_con-rbd+nmci-rbd+mci/testset/y_fold_%d.npy' % (fold + 1))

    y_predict = model.predict(x_test)

    model_wo_softmax = innvestigate.utils.model_wo_softmax(model)  # model without softmax output
    analyzer = innvestigate.create_analyzer("deep_taylor", model_wo_softmax)  # XAI method

    LRP_result = np.zeros(np.shape(x_test))
    for m in range(np.size(x_test, 0)):
        analysed_results = analyzer.analyze(np.reshape(x_test[m, :, :, :, :], (1, 67, 67, 14, 1)))
        LRP_result[m, :, :, :, :] = analysed_results

    pred = np.argmax(y_predict, axis=1)
    label = np.argmax(y_test, axis=1)
    compare = np.zeros(shape=(np.size(pred, axis=0), 1))
    for p in range(np.size(pred, axis=0)):
        if pred[p] == label[p]:
            compare[p] = 0
        elif pred[p] != label[p]:
            compare[p] = 1
    compare_index = np.where(compare != 0)
    incorrect_index = np.array(compare_index)[0, :]
    correct_index_label = np.delete(label, incorrect_index)
    correct_index = np.delete(np.arange(np.size(x_test, axis=0)), incorrect_index)
    correct_LRP = np.squeeze(LRP_result[correct_index, :, :, :, :])

    # LRP_con_index = np.squeeze(np.array(np.where(correct_index_label == 0)))
    # LRP_con = np.mean(correct_LRP[LRP_con_index, :, :, :], axis=0)
    LRP_rbd_nmci_index = np.squeeze(np.array(np.where(correct_index_label == 1)))
    LRP_rbd_nmci = np.mean(correct_LRP[LRP_rbd_nmci_index, :, :, :], axis=0)
    LRP_rbd_mci_index = np.squeeze(np.array(np.where(correct_index_label == 2)))
    LRP_rbd_mci = np.mean(correct_LRP[LRP_rbd_mci_index, :, :, :], axis=0)

    # save numpy_array to .mat file
    np.save('D:/ANT_3D_CNN_con-rbd+nmci-rbd+mci/LRP/LRP_nmci_fold_%d.npy' % (fold + 1), LRP_rbd_nmci)
    np.save('D:/ANT_3D_CNN_con-rbd+nmci-rbd+mci/LRP/LRP_mci_fold_%d.npy' % (fold + 1), LRP_rbd_mci)


terminate_time = timeit.default_timer()         # 종료 시간 체크
print("%f초 걸렸습니다." % (terminate_time - start_time))     # 딥러닝 돌아가는데 걸린 시간
