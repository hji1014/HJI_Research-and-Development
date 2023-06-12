from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization
#from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization, LeakyReLU
from keras.layers.advanced_activations import LeakyReLU
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

all_PT_acc = []
all_TL_acc = []
all_PT_acc_train = []
all_TL_acc_train = []

# load data #
data1 = mat73.loadmat('D:/ANT_3D_CNN/ANT_source_DL/control/ANT_con_source_3d_normalization.mat')
data2 = mat73.loadmat('D:/ANT_3D_CNN/ANT_source_DL/rbd/ANT_rbd_source_3d_normalization.mat')
con_1000 = data1['ANT_con_source_3d_normalization']
rbd_1000 = data2['ANT_rbd_source_3d_normalization']
con_label = np.zeros(shape=(np.size(con_1000, axis=0), 1))
rbd_label = np.ones(shape=(np.size(rbd_1000, axis=0), 1))
gt_con = data1['ground_truth']
gt_rbd = data2['ground_truth']
del data1, data2

#for TL_rbd_num in range(int(max(gt_rbd))):                          # TL : Transfer Learning, PT : Pre-Training
for TL_rbd_num in range(40, 62):  # TL : Transfer Learning, PT : Pre-Training

    sub = TL_rbd_num + 1

    while True:

        print("\n 현재 pre-training 중... :")
        print("\n 현재 test 환자 번호는 :", sub)

        # 62명 각각 fine tuning test
        rbd_TL_idx = np.where(gt_rbd == (TL_rbd_num + 1))
        rbd_PT_idx = np.where(gt_rbd != (TL_rbd_num + 1))
        rbd_PT = np.squeeze(rbd_1000[rbd_PT_idx, :, :, :])
        rbd_TL = np.squeeze(rbd_1000[rbd_TL_idx, :, :, :])
        rbd_PT_label = np.reshape(rbd_label[rbd_PT_idx, :], (-1, 1))
        rbd_TL_label = np.reshape(rbd_label[rbd_TL_idx, :], (-1, 1))

        # NC는 전부 섞어서 test 환자 trial 수만큼 떼어 놓고 나머지 pre-training 하기.
        con_shuffle_idx = np.random.choice(int(np.size(gt_con, axis=0)), int(np.size(gt_con, axis=0)), replace=False)
        con_TL_idx = con_shuffle_idx[:np.size(rbd_TL_label, axis=0)]
        con_PT_idx = con_shuffle_idx[np.size(rbd_TL_label, axis=0):]
        con_TL = con_1000[con_TL_idx, :, :, :]
        con_PT = con_1000[con_PT_idx, :, :, :]
        con_TL_label = con_label[con_TL_idx, :]
        con_PT_label = con_label[con_PT_idx, :]

        # pre-training data 불균형 문제 해결
        PT_balance_idx = np.sort(np.random.choice(np.size(rbd_PT, axis=0), np.size(con_PT, axis=0), replace=False))
        rbd_PT = rbd_PT[PT_balance_idx, :, :, :]
        rbd_PT_label = rbd_PT_label[PT_balance_idx, :]
        con_rbd_PT = np.concatenate((con_PT, rbd_PT), axis=0)
        con_rbd_PT = np.nan_to_num(con_rbd_PT)
        con_rbd_PT = np.reshape(con_rbd_PT, (-1, 60, 120, 15, 1))
        con_rbd_PT_label = np.concatenate((con_PT_label, rbd_PT_label), axis=0)

        # Pre-training (PT)
        x_train_PT, x_val_PT, y_train_PT, y_val_PT = train_test_split(con_rbd_PT, con_rbd_PT_label, random_state=seed, shuffle=True, test_size=0.1)

        del rbd_PT_idx, con_PT_idx, rbd_PT, rbd_PT_label, con_PT, con_PT_label, PT_balance_idx, con_rbd_PT, con_rbd_PT_label

        premodel = Sequential()
        premodel.add(Conv3D(filters=16, kernel_size=(3, 3, 2), kernel_initializer='he_normal', input_shape=(60, 120, 15, 1), activation='relu'))
        premodel.add(Conv3D(filters=32, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
        premodel.add(MaxPooling3D(pool_size=2))
        premodel.add(Dropout(0.25))  # turn off of 25% nodes
        premodel.add(Conv3D(filters=16, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
        premodel.add(Conv3D(filters=32, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
        premodel.add(MaxPooling3D(pool_size=2))
        premodel.add(Dropout(0.25))
        premodel.add(Flatten())
        premodel.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
        premodel.add(Dropout(0.5))
        premodel.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
        premodel.add(Dropout(0.5))
        premodel.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))
        premodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        premodel.summary()

        # premodel = Sequential()
        # premodel.add(Conv3D(filters=16, kernel_size=(3, 3, 2), kernel_initializer='he_normal', input_shape=(60, 120, 15, 1)))
        # premodel.add(LeakyReLU(alpha=0.3))
        # premodel.add(Conv3D(filters=32, kernel_size=(3, 3, 2), kernel_initializer='he_normal',))
        # premodel.add(LeakyReLU(alpha=0.3))
        # premodel.add(MaxPooling3D(pool_size=2))
        # premodel.add(Dropout(0.25))  # turn off of 25% nodes
        # premodel.add(Conv3D(filters=16, kernel_size=(3, 3, 2), kernel_initializer='he_normal'))
        # premodel.add(LeakyReLU(alpha=0.3))
        # premodel.add(Conv3D(filters=32, kernel_size=(3, 3, 2), kernel_initializer='he_normal'))
        # premodel.add(LeakyReLU(alpha=0.3))
        # premodel.add(MaxPooling3D(pool_size=2))
        # premodel.add(Dropout(0.25))
        # premodel.add(Flatten())
        # premodel.add(Dense(256, kernel_initializer='he_normal'))
        # premodel.add(LeakyReLU(alpha=0.3))
        # premodel.add(Dropout(0.5))
        # premodel.add(Dense(128, kernel_initializer='he_normal'))
        # premodel.add(LeakyReLU(alpha=0.3))
        # premodel.add(Dropout(0.5))
        # premodel.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))
        # premodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # premodel.summary()

        early_stopping_callback1 = EarlyStopping(monitor='loss', patience=5)
        tensorboard1 = TensorBoard(log_dir="D:/ANT_3D_CNN_source_transfer2/logs_PT/{}".format(time()))

        history1 = premodel.fit(x_train_PT, y_train_PT, epochs=100, batch_size=32, verbose=2, callbacks=[early_stopping_callback1, tensorboard1])
        #history1 = premodel.fit(x_train_PT, y_train_PT, epochs=100, batch_size=64, verbose=2, callbacks=[early_stopping_callback1])

        PT_acc = "%.8f" % (premodel.evaluate(x_val_PT, y_val_PT)[1])
        print("\n 현재 pre-training accuracy :", PT_acc)
        PT_acc_train = "%.8f" % (premodel.evaluate(x_train_PT, y_train_PT)[1])
        print("\n 현재 pre-training 학습 accuracy :", PT_acc_train)

        PT_acc_train_float = float(PT_acc_train) * 100

        if PT_acc_train_float > 80:
            all_PT_acc.append(PT_acc)
            all_PT_acc_train.append(PT_acc_train)
            break
        else:
            print('******************************************************')
            print('******************** 재학습 진행 중 ********************')
            print('******************************************************')
            del x_train_PT, x_val_PT, y_train_PT, y_val_PT

    # 모델 weights 저장
    premodel.save_weights('D:/ANT_3D_CNN_source_transfer2/premodel_weights/premodel_weights_sub_%d.h5' % sub)
    np.save('D:/ANT_3D_CNN_source_transfer2/premodel_testset/premodel_x_sub_%d.npy' % sub, x_val_PT)
    np.save('D:/ANT_3D_CNN_source_transfer2/premodel_testset/premodel_y_sub_%d.npy' % sub, y_val_PT)

    del x_train_PT, x_val_PT, y_train_PT, y_val_PT

    # transfer learning data 불균형 문제 해결
    if np.size(con_TL, axis=0) > np.size(rbd_TL, axis=0):
        print('con_TL > rbd_TL')
        print('불균형 문제 해결 중...')
        TL_balance_idx = np.sort(np.random.choice(np.size(con_TL, axis=0), np.size(rbd_TL, axis=0), replace=False))
        con_TL = con_TL[TL_balance_idx, :, :, :]
        con_TL_label = con_TL_label[TL_balance_idx, :]
    elif np.size(con_TL, axis=0) == np.size(rbd_TL, axis=0):
        print('TL data 불균형 문제 없음')
    else:
        print('con_TL < rbd_TL')
        print('불균형 문제 해결 중...')
        TL_balance_idx = np.sort(np.random.choice(np.size(rbd_TL, axis=0), np.size(con_TL, axis=0), replace=False))
        rbd_TL = rbd_TL[TL_balance_idx, :, :, :]
        rbd_TL_label = rbd_TL_label[TL_balance_idx, :]

    con_rbd_TL = np.concatenate((con_TL, rbd_TL), axis=0)
    con_rbd_TL = np.nan_to_num(con_rbd_TL)
    con_rbd_TL = np.reshape(con_rbd_TL, (-1, 60, 120, 15, 1))
    con_rbd_TL_label = np.concatenate((con_TL_label, rbd_TL_label), axis=0)

    idx = np.arange(con_rbd_TL_label.shape[0])  # data shuffling
    np.random.shuffle(idx)

    con_rbd_TL = con_rbd_TL[idx, :, :, :, :]
    con_rbd_TL_label = con_rbd_TL_label[idx, :]

    # pre-train 성능 확인
    #PT_acc2 = "%.8f" % (premodel.evaluate(con_rbd_TL, con_rbd_TL_label)[1])
    #print("\n premodel accuracy :", PT_acc2)
    #PT_predict = premodel.predict(con_rbd_TL)

    # Transfer learning (TL) 5-fold CV

    x_train_TL, x_test_TL, y_train_TL, y_test_TL = train_test_split(con_rbd_TL, con_rbd_TL_label, random_state=seed, shuffle=True, test_size=0.2)

    # for i in premodel.layers[:11]:
    #     i.trainable = False
    # premodel.summary()
    for i in premodel.layers[:9]:      #best 9, 11 (?)
        i.trainable = False
    premodel.summary()

    transfer_model = Sequential(premodel.layers[:14])
    Adam = optimizers.Adam(learning_rate=0.001)
    transfer_model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    transfer_model.summary()

    early_stopping_callback2 = EarlyStopping(monitor='loss', patience=10)
    tensorboard2 = TensorBoard(log_dir="D:/ANT_3D_CNN_source_transfer2/logs_TL/{}".format(time()))

    history2 = transfer_model.fit(x_train_TL, y_train_TL, epochs=100, batch_size=32, verbose=2, callbacks=[early_stopping_callback2, tensorboard2])

    TL_acc = "%.8f" % (transfer_model.evaluate(x_test_TL, y_test_TL)[1])
    print("\n Tranfer learning 현재 fold accuracy :", TL_acc)
    all_TL_acc.append(TL_acc)
    TL_acc_train = "%.8f" % (transfer_model.evaluate(x_train_TL, y_train_TL)[1])
    print("\n Tranfer learning 현재 fold 학습 accuracy :", TL_acc_train)
    all_TL_acc_train.append(TL_acc_train)

    transfer_model.save('D:/ANT_3D_CNN_source_transfer2/transfer_model/transfer_model_sub_%d.h5' % sub)
    transfer_model.save_weights('D:/ANT_3D_CNN_source_transfer2/transfer_model_weights/transfer_model_weights_sub_%d.h5' % sub)
    np.save('D:/ANT_3D_CNN_source_transfer2/transfer_model_testset/transfer_model_x_sub_%d.npy' % sub, x_test_TL)
    np.save('D:/ANT_3D_CNN_source_transfer2/transfer_model_testset/transfer_model_y_sub_%d.npy' % sub, y_test_TL)
    # f += 1

    del premodel, transfer_model, x_train_TL, x_test_TL, y_train_TL, y_test_TL

all_PT_acc_train = np.array(all_PT_acc_train)
all_PT_acc_train = all_PT_acc_train.astype(np.double) * 100
all_PT_acc = np.array(all_PT_acc)
all_PT_acc = all_PT_acc.astype(np.double) * 100
all_TL_acc_train = np.array(all_TL_acc_train)
all_TL_acc_train = all_TL_acc_train.astype(np.double) * 100
all_TL_acc = np.array(all_TL_acc)
all_TL_acc = all_TL_acc.astype(np.double) * 100

# np.save('D:/ANT_3D_CNN_source_transfer/premodel_accuracy/all_PT_acc.npy', all_PT_acc)
# np.save('D:/ANT_3D_CNN_source_transfer/transfer_model_accuracy/all_TL_acc.npy', all_TL_acc)
# np.save('D:/ANT_3D_CNN_source_transfer/premodel_accuracy/all_PT_acc_train.npy', all_PT_acc_train)
# np.save('D:/ANT_3D_CNN_source_transfer/transfer_model_accuracy/all_TL_acc_train.npy', all_TL_acc_train)

terminate_time = timeit.default_timer()         # 종료 시간 체크
print("%f초 걸렸습니다." % (terminate_time - start_time))     # 딥러닝 돌아가는데 걸린 시간

avg_pt_train = np.mean(all_PT_acc_train)
std_pt_train = np.std(all_PT_acc_train)
avg_pt = np.mean(all_PT_acc)
std_pt = np.std(all_PT_acc)
avg_tl_train = np.mean(all_TL_acc_train)
std_tl_train = np.std(all_TL_acc_train)
avg_tl = np.mean(all_TL_acc)
std_tl = np.std(all_TL_acc)

"""notification to discord"""
# -------------------------------------------------------------------------------------------------------------------------------------------------
from knockknock import discord_sender
webhook_url = "https://discord.com/api/webhooks/1014081244734173274/kCGlk4rXRPb4LSOdf4ECz9P0lvbiHr5tq4EOR2Zf6RPd8OgU8eytgtG2-IjER1abFt4y"
@discord_sender(webhook_url=webhook_url)
def DL_notification():
    #import time
    #time.sleep(3)
    return {'averaged test acc of premodel' : avg_pt}, {'averaged test acc of transfer model' : avg_tl}, {'소요시간' :(terminate_time - start_time)} # Optional return value
DL_notification()
# -------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------정확도 계산-------------------------------------------------------------------------

for sub in range(62):
    print('---------------- 현재 정확도 계산 중인 피험자 번호 :', (sub + 1), '----------------')

    transfer_model = Sequential()
    transfer_model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), kernel_initializer='he_normal', input_shape=(60, 120, 15, 1), activation='relu'))
    transfer_model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
    transfer_model.add(MaxPooling3D(pool_size=2))
    transfer_model.add(Dropout(0.25))  # turn off of 25% nodes
    transfer_model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
    transfer_model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
    transfer_model.add(MaxPooling3D(pool_size=2))
    transfer_model.add(Dropout(0.25))
    transfer_model.add(Flatten())
    transfer_model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
    transfer_model.add(Dropout(0.5))
    transfer_model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
    transfer_model.add(Dropout(0.5))
    transfer_model.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))
    transfer_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    transfer_model.summary()
    #transfer_model.load_weights('D:/ANT_3D_CNN_source_transfer2/premodel_weights/premodel_weights_sub_%d.h5' % (sub + 1))
    transfer_model.load_weights('D:/ANT_3D_CNN_source_transfer2/transfer_model_weights/transfer_model_weights_sub_%d.h5' % (sub + 1))

    # x_test_PT = np.load('D:/ANT_3D_CNN_source_transfer/premodel_testset/premodel_x_sub_%d.npy' % (sub + 1))
    # y_test_PT = np.load('D:/ANT_3D_CNN_source_transfer/premodel_testset/premodel_y_sub_%d.npy' % (sub + 1))
    x_test_TL = np.load('D:/ANT_3D_CNN_source_transfer2/transfer_model_testset/transfer_model_x_sub_%d.npy' % (sub + 1))
    y_test_TL = np.load('D:/ANT_3D_CNN_source_transfer2/transfer_model_testset/transfer_model_y_sub_%d.npy' % (sub + 1))

    # PT_test_acc = "%.4f" % (transfer_model.evaluate(x_test_PT, y_test_PT)[1])
    TL_test_acc = "%.4f" % (transfer_model.evaluate(x_test_TL, y_test_TL)[1])
    # print("\n 현재 피험자 Pre-train test accuracy :", PT_test_acc)
    print("\n 현재 피험자 Transfer learning test accuracy :", TL_test_acc)

    # all_PT_acc.append(PT_test_acc)
    all_TL_acc.append(TL_test_acc)

# all_PT_acc_train = np.array(all_PT_acc_train)
# all_PT_acc_train = all_PT_acc_train.astype(np.double) * 100
# all_PT_acc = np.array(all_PT_acc)
# all_PT_acc = all_PT_acc.astype(np.double) * 100
# all_TL_acc_train = np.array(all_TL_acc_train)
# all_TL_acc_train = all_TL_acc_train.astype(np.double) * 100
all_TL_acc = np.array(all_TL_acc)
all_TL_acc = all_TL_acc.astype(np.double) * 100

# np.save('D:/ANT_3D_CNN_source_transfer/premodel_accuracy/all_PT_acc.npy', all_PT_acc)
np.save('D:/ANT_3D_CNN_source_transfer2/transfer_model_accuracy/all_TL_acc.npy', all_TL_acc)
# np.save('D:/ANT_3D_CNN_source_transfer/premodel_accuracy/all_PT_acc_train.npy', all_PT_acc_train)
# np.save('D:/ANT_3D_CNN_source_transfer/transfer_model_accuracy/all_TL_acc_train.npy', all_TL_acc_train)

a = np.mean(all_TL_acc)