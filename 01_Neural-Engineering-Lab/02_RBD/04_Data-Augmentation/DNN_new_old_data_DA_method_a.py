# all-shift

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import h5py
import random
import timeit

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다. #

seed = 0
np.random.seed(seed)
start_time = timeit.default_timer()             # 시작 시간 체크

# load data #

data1 = scipy.io.loadmat('./data_augmentation/data/con_o_2.mat')
data2 = scipy.io.loadmat('./data_augmentation/data/rbd_o_2.mat')
data3 = scipy.io.loadmat('./data_augmentation/data/R_G_all.mat')    # ground truth

# data 가공 #

con_1000 = data1['con'].T                                # con/rbd 각 1000 trials
rbd_1000 = data2['rbd'].T
con_label = np.zeros(shape=(9242, 1))
rbd_label = np.ones(shape=(7929, 1))
ground_truth = data3['R_G_all']

con_rbd = np.concatenate((con_1000, rbd_1000), axis=0)
con_rbd_label = np.concatenate((con_label, rbd_label), axis=0)

# subject당 10% 추출 #

sub_num = np.max(ground_truth[:, 1])
rand_percent = 0.1  # 10%
#rand_select_data = []
rand_select_data = np.zeros((1, 16800))
rand_select_data_label = np.zeros((1, 1))

for i in range(sub_num):
    sub_index = np.where(ground_truth[:, 1] == i)
    sub_index = np.array(sub_index)
    sub_index = np.reshape(sub_index, (np.size(sub_index), ))
    random_choice = np.random.choice(sub_index, round(np.size(sub_index) * rand_percent), replace=False)
    random_choice = np.sort(random_choice)

    sub_data = con_rbd[random_choice, :]
    sub_data_label = con_rbd_label[random_choice, :]

    rand_select_data = np.vstack([rand_select_data, sub_data])
    rand_select_data_label = np.vstack([rand_select_data_label, sub_data_label])

zz = rand_select_data[1:, :]                    # input
zzz = rand_select_data_label[1:, :]             # label
zzzz = np_utils.to_categorical(zzz)             # label for softmax

# train/validation #

#============================= DA configuration ==============================

DA = True                  # data augmentation 실행여부 설정
aug_times = 3               # data augmentation-times
middle_time = 46            # 중간점:225ms

#==============================================================================

# k-fold C.V. #
n_fold = 5
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

accuracy = []

for train, test in skf.split(zz, zzz):
    # model construction #

    x = layers.Input(shape=(16800,))
    h1 = layers.Activation('relu')(layers.Dense(300, name='zz')(x))
    h2 = layers.Activation('relu')(layers.Dense(200)(h1))
    h3 = layers.Activation('relu')(layers.Dense(100)(h2))
    y = layers.Activation('softmax')(layers.Dense(2)(h3))

    model = models.Model(x, y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    x_train = zz[train, :]
    y_train = zzzz[train, :]

    if DA:

        n = x_train.copy()
        p = x_train.copy()
        m = x_train.copy()
        p2 = x_train.copy()
        m2 = x_train.copy()
        p3 = x_train.copy()
        m3 = x_train.copy()

        for i in range(len(x_train[:, 0])):
            for j in range(60):
                # data augmentation range : 25ms~425ms
                t = middle_time + 280 * j
                DA_range = x_train[i, t - 140:t + 140]
                DA_d_n = DA_range.copy() * 1
                DA_d_p = DA_range.copy() * 1
                DA_d_m = DA_range.copy() * 1

                n[i, t - 140:t + 140] = DA_d_n
                p[i, t - 138:t + 140] = DA_d_p[:278]
                p[i, t - 140:t - 138] = 0
                m[i, t - 140:t + 138] = DA_d_m[:278]
                m[i, t + 138:t + 140] = 0

        # 3-times DA
        if aug_times == 3:
            x_train_aug = np.concatenate((n, p, m), axis=0)
            y_train_aug = np.concatenate((y_train, y_train, y_train), axis=0)

    #early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)
    #history = model.fit(x_train, y_train, epochs=3000, batch_size=5, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping_callback])
    #history = model.fit(x_train, y_train, epochs=20, batch_size=5, validation_data=(x_val, y_val), verbose=2)
    history = model.fit(x_train_aug, y_train_aug, epochs=20, batch_size=5, verbose=2)

    #print("\n Test Accuracy : %.4f" % (model.evaluate(zz[test], zzzz[test], verbose=0)[1]))          # model.predict()로 test data 예측 결과 눈으로 볼 수 있음
    #print("\n Test loss : %.4f" % (model.evaluate(zz[test], zzzz[test])[0]))

    k_accuracy = "%.4f" % (model.evaluate(zz[test], zzzz[test])[1])
    accuracy.append(k_accuracy)

print("\n %.f fold accuracy : " % n_fold, accuracy)
np_acc = np.array(accuracy)
np_acc = np_acc.astype(np.double)
avg_acc = np.mean(np_acc)
print(" %.f fold 평균 정확도 : " % n_fold, avg_acc)

terminate_time = timeit.default_timer()         # 종료 시간 체크
print("%f초 걸렸습니다." % (terminate_time - start_time))     # 딥러닝 돌아가는데 걸린 시간


# plotting
# acc= history.history['accuracy']
# val_acc= history.history['val_accuracy']
# y_vloss = history.history['val_loss']
# y_loss = history.history['loss']
#
# x_len = np.arange(len(y_loss))
# plt.plot(x_len, acc, marker='.', c="red", label='Train_acc')
# plt.plot(x_len, val_acc, marker='.', c="lightcoral", label='Validation_acc')
# plt.plot(x_len, y_vloss, marker='.', c="cornflowerblue", label='Validation_loss')
# plt.plot(x_len, y_loss, marker='.', c="blue", label='Train_loss')
#
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss/acc')
# plt.show()
