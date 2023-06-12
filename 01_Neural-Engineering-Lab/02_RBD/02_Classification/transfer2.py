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

pretrainset = np.nan_to_num(pretrainset)                                # nan 제거 및 모델에 들어갈 shape 로 변경
pretrainset = np.reshape(pretrainset, (-1, 67, 67, 14, 1))

# transfer learning 할 iRBD 피험자
subject_num = 25    # 25~62
rbd_test_25_idx = np.where(gt_rbd_test == subject_num)
rbd_test_25 = np.squeeze(rbd_test[rbd_test_25_idx, :, :, :])
rbd_test_25 = np.nan_to_num(rbd_test_25)
rbd_test_25 = np.reshape(rbd_test_25, (-1, 67, 67, 14, 1))
rbd_test_label_25 = np.squeeze(rbd_test_label[rbd_test_25_idx, :])
rbd_test_label_25_one_hot_encoding = np_utils.to_categorical(rbd_test_label_25)

x_train, x_val, y_train, y_val = train_test_split(pretrainset, pretrainset_label_one_hot_encoding, random_state=seed, shuffle=True, test_size=0.1)
x_train_transfer, x_test, y_train_transfer, y_test = train_test_split(rbd_test_25, rbd_test_label_25_one_hot_encoding, random_state=seed, shuffle=True, test_size=0.8)

# pretrained model 불러오기
pretrained_model = load_model('pretrained_model.h5')
# 마지막 dense layer 두 개(128, 2) 빼고 frozen
pretrained_model.summary()
for i in pretrained_model.layers[:11]:
    i.trainable = False
pretrained_model.summary()

transfer_model = Sequential(pretrained_model.layers[:14])
Adam = optimizers.Adam(learning_rate=0.0001)
transfer_model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
transfer_model.summary()

early_stopping_callback = EarlyStopping(monitor='loss', patience=10)
history = transfer_model.fit(x_train_transfer, y_train_transfer, epochs=100, batch_size=8, verbose=2, callbacks=[early_stopping_callback])

#k_accuracy = "%.8f" % (transfer_model.evaluate(x_test, y_test)[1])
#print("\n 현재 fold validation accuracy :", k_accuracy)
#predict_test = transfer_model.predict(x_test)
#predict_train_transfer = transfer_model.predict(x_train_transfer)
#predict_val = transfer_model.predict(x_val)

# pretrained model의 validation에 사용된 con data 섞어서 test 진행
y_val_con_idx = np.where(y_val[:, 0] == 1)
y_val_con = np.squeeze(y_val[y_val_con_idx, :])
x_val_con = x_val[y_val_con_idx, :, :, :, :]
x_val_con = np.reshape(x_val_con, (-1, 67, 67, 14, 1))

test_num = np.size(y_test, axis=0)
con_test_num_random = np.sort(np.random.choice(np.size(y_val_con, axis=0), test_num, replace=False))

y_val_con = y_val_con[con_test_num_random, :]
x_val_con = x_val_con[con_test_num_random, :, :, :, :]

new_test_x = np.concatenate((x_val_con, x_test), axis=0)
new_test_y = np.concatenate((y_val_con, y_test), axis=0)

idx = np.arange(new_test_y.shape[0])                # data shuffling
np.random.shuffle(idx)

new_test_x = new_test_x[idx, :, :, :, :]
new_test_y = new_test_y[idx, :]

k_accuracy2 = "%.8f" % (transfer_model.evaluate(new_test_x, new_test_y)[1])
print("\n 현재 fold validation accuracy :", k_accuracy2)
predict_multi = transfer_model.predict(new_test_x)

# confusion matrix
from sklearn.metrics import confusion_matrix
a = predict_multi[:, 1]
b = np.where(a>=0.5, 1, 0)
c = new_test_y[:, 1]
cf = confusion_matrix(c, b)
