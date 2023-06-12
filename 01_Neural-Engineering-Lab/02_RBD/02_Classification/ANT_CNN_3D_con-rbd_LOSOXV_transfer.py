from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization, Activation
from keras.utils import np_utils
from keras import optimizers
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
con_1000 = data1['ANT_con_topo_3d']
rbd_1000 = data2['ANT_rbd_topo_3d']
gt_con = data1['ground_truth']
gt_rbd = data2['ground_truth']
del data1, data2

train_rbd_idx = np.where(gt_rbd <= 24)          # train 들어가는 rbd (1~24번)
test_rbd_idx = np.where(gt_rbd > 24)            # test 들어가는 rbd (25~62번)

rbd_train = np.squeeze(rbd_1000[train_rbd_idx, :, :, :])
rbd_test = np.squeeze(rbd_1000[test_rbd_idx, :, :, :])
gt_rbd_train = gt_rbd[train_rbd_idx]
gt_rbd_test = gt_rbd[test_rbd_idx]
con_train = con_1000
del con_1000, rbd_1000, gt_rbd, train_rbd_idx, test_rbd_idx

con_train_label = np.zeros(shape=(np.size(con_train, axis=0), 1))
rbd_train_label = np.ones(shape=(np.size(rbd_train, axis=0), 1))
rbd_test_label = np.ones(shape=(np.size(rbd_test, axis=0), 1))

pretrainset = np.concatenate((con_train, rbd_train), axis=0)
pretrainset_label = np.concatenate((con_train_label, rbd_train_label), axis=0)
pretrainset_label_one_hot_encoding = np_utils.to_categorical(pretrainset_label)
del con_train, rbd_train, con_train_label, rbd_train_label

# sf = np.random.choice(int(np.size(pretrainset, axis=0)), int(np.size(pretrainset, axis=0)), replace=False)        # data shuffling
# pretrainset = pretrainset[sf, :, :, :]
# pretrainset_label = pretrainset_label[sf, :]
# pretrainset_label_one_hot_encoding = pretrainset_label_one_hot_encoding[sf, :]

pretrainset = np.nan_to_num(pretrainset)                                # nan 제거 및 모델에 들어갈 shape 로 변경
pretrainset = np.reshape(pretrainset, (-1, 67, 67, 14, 1))

x_train, x_val, y_train, y_val = train_test_split(pretrainset, pretrainset_label_one_hot_encoding, random_state=seed, shuffle=True, test_size=0.1)

pretrained_model = Sequential()
pretrained_model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), input_shape=(67, 67, 14, 1), activation='relu'))
pretrained_model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), activation='relu'))
pretrained_model.add(MaxPooling3D(pool_size=2))
pretrained_model.add(Dropout(0.25))  # turn off of 25% nodes
pretrained_model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), activation='relu'))
pretrained_model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), activation='relu'))
pretrained_model.add(MaxPooling3D(pool_size=2))
pretrained_model.add(Dropout(0.25))
pretrained_model.add(Flatten())
pretrained_model.add(Dense(256, activation='relu'))
pretrained_model.add(Dropout(0.5))
pretrained_model.add(Dense(128, activation='relu'))
pretrained_model.add(Dropout(0.5))
pretrained_model.add(Dense(2, activation='softmax'))
pretrained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
pretrained_model.summary()

early_stopping_callback = EarlyStopping(monitor='loss', patience=10)
history = pretrained_model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=2, callbacks=[early_stopping_callback])

k_accuracy = "%.8f" % (pretrained_model.evaluate(x_val, y_val)[1])
print("\n 현재 fold validation accuracy :", k_accuracy)
# predict_val = pretrained_model.predict(x_val)
# predict_train = pretrained_model.predict(x_train)
# pretrained_model.save('pretrained_model.h5')

rbd_test_25_idx = np.where(gt_rbd_test == 25)
rbd_test_25 = np.squeeze(rbd_test[rbd_test_25_idx, :, :, :])
rbd_test_25 = np.nan_to_num(rbd_test_25)
rbd_test_25 = np.reshape(rbd_test_25, (-1, 67, 67, 14, 1))
rbd_test_label_25 = np.squeeze(rbd_test_label[rbd_test_25_idx, :])
rbd_test_label_25_one_hot_encoding = np_utils.to_categorical(rbd_test_label_25)

x_train_transfer, x_test, y_train_transfer, y_test = train_test_split(rbd_test_25, rbd_test_label_25_one_hot_encoding, random_state=seed, shuffle=True, test_size=0.8)

# pretrained model 불러오기
pretrained_model = load_model('pretrained_model.h5')
pretrained_model.summary()
for i in pretrained_model.layers[:11]:
    i.trainable = False
pretrained_model.summary()

# transfer_model = Sequential(pretrained_model.layers[:11])
# transfer_model.add(Dense(128, activation='relu', name='transfer_Dense1'))
# transfer_model.add(Dropout(0.5, name='transfer_Dropout1'))
# transfer_model.add(Dense(2, activation='softmax', name='transfer_Dense2'))
# transfer_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# transfer_model.summary()
transfer_model = Sequential(pretrained_model.layers[:14])
Adam = optimizers.Adam(learning_rate=0.0001)
transfer_model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
transfer_model.summary()

early_stopping_callback = EarlyStopping(monitor='loss', patience=10)
history = transfer_model.fit(x_train_transfer, y_train_transfer, epochs=100, batch_size=8, verbose=2, callbacks=[early_stopping_callback])

k_accuracy = "%.8f" % (transfer_model.evaluate(x_test, y_test)[1])
print("\n 현재 fold validation accuracy :", k_accuracy)

predict_test = transfer_model.predict(x_test)
predict_train_transfer = transfer_model.predict(x_train_transfer)
predict_val = transfer_model.predict(x_val)

########################################################### con data 합쳐서 test 진행 #############################################################

con_test_idx = np.where(y_val[:, 0] == 1)
con_test = x_val[con_test_idx, :, :, :, :]
con_test = np.reshape(con_test, (-1, 67, 67, 14, 1))
con_test_label = y_val[con_test_idx, :]
con_test_label = np.squeeze(con_test_label)
con_test_num = np.size(y_test, axis=0)
x_test_multi = np.concatenate((con_test[0:con_test_num, :, :, :, :], x_test), axis=0)
y_test_multi = np.concatenate((con_test_label[0:con_test_num, :], y_test), axis=0)

k_accuracy_multi = "%.8f" % (transfer_model.evaluate(x_test_multi, y_test_multi)[1])
print("\n 현재 fold validation accuracy :", k_accuracy_multi)
predict_test_multi = transfer_model.predict(x_test_multi)

################################################################### con data 합쳐서 fine-tunning 진행 #########################################################

con_test_idx = np.where(y_val[:, 0] == 1)
con_test = x_val[con_test_idx, :, :, :, :]
con_test = np.reshape(con_test, (-1, 67, 67, 14, 1))
con_test_label = y_val[con_test_idx, :]
con_test_label = np.squeeze(con_test_label)
con_test_num = np.size(y_train_transfer, axis=0)
x_train_transfer_multi = np.concatenate((con_test[0:con_test_num, :, :, :, :], x_train_transfer), axis=0)
y_train_transfer_multi = np.concatenate((con_test_label[0:con_test_num, :], y_train_transfer), axis=0)

early_stopping_callback = EarlyStopping(monitor='loss', patience=10)
history = transfer_model.fit(x_train_transfer_multi, y_train_transfer_multi, epochs=10, batch_size=32, verbose=2, callbacks=[early_stopping_callback])

k_accuracy_multi = "%.8f" % (transfer_model.evaluate(x_test_multi, y_test_multi)[1])
print("\n 현재 fold validation accuracy :", k_accuracy_multi)
predict_test_multi = transfer_model.predict(x_test_multi)
