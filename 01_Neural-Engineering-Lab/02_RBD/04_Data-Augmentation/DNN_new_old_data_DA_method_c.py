# peak-shift

from keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from keras.callbacks import EarlyStopping
import keras
from keras.utils import np_utils
from keras import layers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import h5py
import random


seed = 0                    # 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
np.random.seed(seed)

# load data #
data1 = scipy.io.loadmat('./data_augmentation/data/con_o_2.mat')
data2 = scipy.io.loadmat('./data_augmentation/data/rbd_o_2.mat')

con_1000 = data1['con']                                # con/rbd 각 1000 trials
rbd_1000 = data2['rbd']
con_1000 = con_1000.T
rbd_1000 = rbd_1000.T
con_label = np.zeros(shape=(9242, 1))
rbd_label = np.ones(shape=(7929, 1))
con = np.concatenate((con_1000, con_label), axis=1)
rbd = np.concatenate((rbd_1000, rbd_label), axis=1)

# data 가공 #
con_rbd = np.concatenate((con, rbd), axis=0)
con_rbd_shuffle = np.random.permutation(con_rbd)

# 일부 추출 #

z = np.random.choice(17171, 2000, replace=False)          # con or rbd 3000개 무작위로 뽑기
z_random = np.sort(z)
zz = con_rbd_shuffle[z_random, :]

# train/validation #

#x_train, x_test, y_train, y_test = train_test_split(con_rbd_shuffle[:, 0:16800], con_rbd_shuffle[:, 16800], random_state=seed, shuffle=True, test_size=0.4)
x_train, x_test, y_train, y_test = train_test_split(zz[:, 0:16800], np_utils.to_categorical(zz[:, 16800]), random_state=seed, shuffle=True, test_size=0.4)
#x_train, x_test, y_train, y_test = train_test_split(zz[:, 0:16800], zz[:, 16800], random_state=seed, shuffle=True, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=seed, shuffle=True, test_size=0.5)

# data augmentation - method(d) #

#============================= DA configuration ==============================

DA = True                  # data augmentation 실행여부 설정
aug_times = 3               # data augmentation-times
middle_time = 46            # 중간점:225ms

#==============================================================================

if DA:

    train_len = len(x_train[:, 0])

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
            DA_range = x_train[i, t - 40:t + 40]
            DA_d_n = DA_range.copy() * 1
            DA_d_p = DA_range.copy() * 1
            DA_d_m = DA_range.copy() * 1

            n[i, t-40:t+40] = DA_d_n
            p[i, t-38:t+42] = DA_d_p
            p[i, t-40:t-38] = 0
            m[i, t-42:t+38] = DA_d_m
            m[i, t+38:t+40] = 0



    # 3-times DA
    if aug_times == 3:
        x_train_aug = np.concatenate((n, p, m), axis=0)
        y_train_aug = np.concatenate((y_train, y_train, y_train), axis=0)


# model construction #

x = layers.Input(shape=(16800,))
h1 = layers.Activation('relu')(layers.Dense(300, name='zz')(x))
h2 = layers.Activation('relu')(layers.Dense(200)(h1))
h3 = layers.Activation('relu')(layers.Dense(100)(h2))
y = layers.Activation('softmax')(layers.Dense(2)(h3))
model = models.Model(x, y)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
#history = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping_callback])
history = model.fit(x_train_aug, y_train_aug, epochs=100, batch_size=5, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping_callback])

print("\n Test Accuracy : %.4f" % (model.evaluate(x_test, y_test, verbose=0)[1]))          # model.predict()로 test data 예측 결과 눈으로 볼 수 있음
print("\n Test loss : %.4f" % (model.evaluate(x_test, y_test)[0]))

acc= history.history['accuracy']
val_acc= history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker='.', c="red", label='Train_acc')
plt.plot(x_len, val_acc, marker='.', c="lightcoral", label='Validation_acc')
plt.plot(x_len, y_vloss, marker='.', c="cornflowerblue", label='Validation_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Train_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()