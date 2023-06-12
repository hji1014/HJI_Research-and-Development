# DNN_new_old_data_DA_method_seg&recombi_time_domain
#
#
#

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras import layers, models
from keras.utils import np_utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import h5py
import random

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다. #

seed = 0
np.random.seed(seed)

# load data #

data1 = scipy.io.loadmat('./data_augmentation/data/con_ori_2d.mat')
data2 = scipy.io.loadmat('./data_augmentation/data/rbd_ori_2d.mat')

# data 가공 #

con_1000 = data1['con']                                # con/rbd 각 1000 trials
rbd_1000 = data2['rbd']
con_label = np.zeros(shape=(9242, 1))
rbd_label = np.ones(shape=(7929, 1))

# data 가공 #
con_rbd = np.concatenate((con_1000, rbd_1000), axis=0)
con_rbd_label = np.concatenate((con_label, rbd_label), axis=0)

# 일부 추출 #

z = np.random.choice(17171, 2000, replace=False)          # con or rbd 3000개 무작위로 뽑기
z_random = np.sort(z)
zz = con_rbd[z_random, :]
zzz = con_rbd_label[z_random, :]                           # input_shape=(60, 280)
zzzz = np_utils.to_categorical(zzz)

# train/validation #

x_train, x_test, y_train, y_test = train_test_split(zz, zzzz, random_state=seed, shuffle=True, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=seed, shuffle=True, test_size=0.5)

# method-seg&recom in time domain #
DA_num = 1200

y_train_con = np.where(y_train[:, 1] == 0)              # train con / rbd separate
y_train_con = list(y_train_con)
y_train_rbd = np.where(y_train[:, 1] == 1)
y_train_rbd = list(y_train_rbd)
x_train_con = x_train[y_train_con, :, :]
x_train_con = x_train_con.reshape(-1, 60, 280)
x_train_rbd = x_train[y_train_rbd, :, :]
x_train_rbd = x_train_rbd.reshape(-1, 60, 280)

con_seg1 = x_train_con[:, :, 0:40]       # <DA of Normal Control> 7segments per 1 trials(1seg=0.2sec)
con_seg2 = x_train_con[:, :, 40:80]
con_seg3 = x_train_con[:, :, 80:120]
con_seg4 = x_train_con[:, :, 120:160]
con_seg5 = x_train_con[:, :, 160:200]
con_seg6 = x_train_con[:, :, 200:240]
con_seg7 = x_train_con[:, :, 240:280]

con_aug = np.zeros((DA_num, 60, 280))

for i in range(DA_num):
    for k in range(60):
        rand_con = np.random.choice(len(x_train_con), 1, replace=False)
        con_s1 = con_seg1[rand_con, k, :]
        del rand_con
        rand_con = np.random.choice(len(x_train_con), 1, replace=False)
        con_s2 = con_seg2[rand_con, k, :]
        del rand_con
        rand_con = np.random.choice(len(x_train_con), 1, replace=False)
        con_s3 = con_seg3[rand_con, k, :]
        del rand_con
        rand_con = np.random.choice(len(x_train_con), 1, replace=False)
        con_s4 = con_seg4[rand_con, k, :]
        del rand_con
        rand_con = np.random.choice(len(x_train_con), 1, replace=False)
        con_s5 = con_seg5[rand_con, k, :]
        del rand_con
        rand_con = np.random.choice(len(x_train_con), 1, replace=False)
        con_s6 = con_seg6[rand_con, k, :]
        del rand_con
        rand_con = np.random.choice(len(x_train_con), 1, replace=False)
        con_s7 = con_seg7[rand_con, k, :]
        del rand_con
        con_recombination = np.concatenate([con_s1, con_s2, con_s3, con_s4, con_s5, con_s6, con_s7], axis=1)   # (1,280)

        con_aug[i, k, :] = con_recombination

rbd_seg1 = x_train_rbd[:, :, 0:40]  # <DA of iRBD patients> 7segments per 1 trials(1seg=0.2sec)
rbd_seg2 = x_train_rbd[:, :, 40:80]
rbd_seg3 = x_train_rbd[:, :, 80:120]
rbd_seg4 = x_train_rbd[:, :, 120:160]
rbd_seg5 = x_train_rbd[:, :, 160:200]
rbd_seg6 = x_train_rbd[:, :, 200:240]
rbd_seg7 = x_train_rbd[:, :, 240:280]

rbd_aug = np.zeros((DA_num, 60, 280))

for i in range(DA_num):
    for k in range(60):
        rand_rbd = np.random.choice(len(x_train_rbd), 1, replace=False)
        rbd_s1 = rbd_seg1[rand_rbd, k, :]
        del rand_rbd
        rand_rbd = np.random.choice(len(x_train_rbd), 1, replace=False)
        rbd_s2 = rbd_seg2[rand_rbd, k, :]
        del rand_rbd
        rand_rbd = np.random.choice(len(x_train_rbd), 1, replace=False)
        rbd_s3 = rbd_seg3[rand_rbd, k, :]
        del rand_rbd
        rand_rbd = np.random.choice(len(x_train_rbd), 1, replace=False)
        rbd_s4 = rbd_seg4[rand_rbd, k, :]
        del rand_rbd
        rand_rbd = np.random.choice(len(x_train_rbd), 1, replace=False)
        rbd_s5 = rbd_seg5[rand_rbd, k, :]
        del rand_rbd
        rand_rbd = np.random.choice(len(x_train_rbd), 1, replace=False)
        rbd_s6 = rbd_seg6[rand_rbd, k, :]
        del rand_rbd
        rand_rbd = np.random.choice(len(x_train_rbd), 1, replace=False)
        rbd_s7 = rbd_seg7[rand_rbd, k, :]
        del rand_rbd
        rbd_recombination = np.concatenate([rbd_s1, rbd_s2, rbd_s3, rbd_s4, rbd_s5, rbd_s6, rbd_s7], axis=1)  # (1,280)

        rbd_aug[i, k, :] = rbd_recombination

x_train_aug = np.concatenate([con_aug, rbd_aug], axis=0)
y_train_aug = np.concatenate([np.zeros((DA_num, 1)), np.ones((DA_num, 1))], axis=0)
y_train_aug_categorical = np_utils.to_categorical(y_train_aug)
x_train_final = np.concatenate([x_train, x_train_aug], axis=0)
y_train_final = np.concatenate([y_train, y_train_aug_categorical], axis=0)

# for CNN input shape #

#x_train = np.reshape(x_train, (-1, 16800, 1))
#x_test = np.reshape(x_test, (-1, 16800, 1))
#x_val = np.reshape(x_val, (-1, 16800, 1))
x_train = x_train.reshape(len(x_train), 16800)            # 파이썬에선 차원치환 필요 없이 reshape 바로 써서 2D->1D변환
x_val = x_val.reshape(len(x_val), 16800)
x_test = x_test.reshape(len(x_test), 16800)
x_train_final = x_train_final.reshape(len(x_train_final), 16800)

#x_train = np.reshape(x_train, (-1, 16800, 1))
#x_train_final = np.reshape(x_train_final, (-1, 16800, 1))
#x_test = np.reshape(x_test, (-1, 16800, 1))
#x_val = np.reshape(x_val, (-1, 16800, 1))




############################################################################################

# model test construction #

x = layers.Input(shape=(16800,))
h1 = layers.Activation('relu')(layers.Dense(300, name='zz')(x))
h2 = layers.Activation('relu')(layers.Dense(200)(h1))
h3 = layers.Activation('relu')(layers.Dense(100)(h2))
y = layers.Activation('softmax')(layers.Dense(2)(h3))
model = models.Model(x, y)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

#early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
#history = model.fit(x_train_final, y_train_final, epochs=100, batch_size=5, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping_callback])
history = model.fit(x_train, y_train, epochs=20, batch_size=5, validation_data=(x_val, y_val), verbose=2)

print("\n Test Accuracy : %.4f" % (model.evaluate(x_test, y_test, verbose=0)[1]))          # model.predict()로 test data 예측 결과 눈으로 볼 수 있음
print("\n Test loss : %.4f" % (model.evaluate(x_test, y_test)[0]))


#############################################################################################



# model 1 construction #

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, input_shape=(16800, 1), activation='relu'))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.summary()

#history = model.fit(x_train, y_train, epochs=20, batch_size=5, validation_data=(x_val, y_val), verbose=2)

#early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
#history = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping_callback])
history = model.fit(x_train_final, y_train_final, epochs=100, batch_size=5, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping_callback])

print("\n Test Accuracy : %.4f" % (model.evaluate(x_test, y_test, verbose=0)[1]))          # model.predict()로 test data 예측 결과 눈으로 볼 수 있음
print("\n Test loss : %.4f" % (model.evaluate(x_test, y_test)[0]))


# model 2 construction #

model2 = Sequential()
model2.add(Conv1D(filters=16, kernel_size=3, input_shape=(16800, 1), activation='relu'))
model2.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model2.add(MaxPooling1D(pool_size=2))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(2, activation='softmax'))
model2.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model2.summary()

history2 = model2.fit(x_train_final, y_train_final, epochs=20, batch_size=5, validation_data=(x_val, y_val), verbose=2)

print("\n Test Accuracy : %.4f" % (model2.evaluate(x_test, y_test, verbose=0)[1]))          # model.predict()로 test data 예측 결과 눈으로 볼 수 있음
print("\n Test loss : %.4f" % (model2.evaluate(x_test, y_test)[0]))

acc= history.history['accuracy']
val_acc= history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.figure(1)
plt.plot(x_len, acc, marker='.', c="red", label='Train_acc')
plt.plot(x_len, val_acc, marker='.', c="lightcoral", label='Validation_acc')
plt.plot(x_len, y_vloss, marker='.', c="cornflowerblue", label='Validation_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Train_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.ylim(0, 2)


acc2= history2.history['accuracy']
val_acc2= history2.history['val_accuracy']
y_vloss2 = history2.history['val_loss']
y_loss2 = history2.history['loss']

x_len2 = np.arange(len(y_loss2))
plt.figure(2)
plt.plot(x_len2, acc2, marker='.', c="red", label='Train_acc')
plt.plot(x_len2, val_acc2, marker='.', c="lightcoral", label='Validation_acc')
plt.plot(x_len2, y_vloss2, marker='.', c="cornflowerblue", label='Validation_loss')
plt.plot(x_len2, y_loss2, marker='.', c="blue", label='Train_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.ylim(0, 2)
plt.show()

print("\n <정확도>")
print("\n Test Accuracy of model 1 : %.4f" % (model.evaluate(x_test, y_test, verbose=0)[1]))
print("\n Test Accuracy of model 2: %.4f" % (model2.evaluate(x_test, y_test, verbose=0)[1]))
