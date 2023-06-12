######################################### LOSOXV example #################################
# import numpy as np
# from sklearn.model_selection import StratifiedGroupKFold
# X = np.ones((17, 2))
# y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
# cv = StratifiedGroupKFold(n_splits=8)
# for train_idxs, test_idxs in cv.split(X, y, groups):
#     print("TRAIN:", groups[train_idxs])
#     print("      ", y[train_idxs])
#     print(" TEST:", groups[test_idxs])
#     print("      ", y[test_idxs])
############################################################################################

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization
from keras.utils import np_utils
import keras_tuner
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
data1 = mat73.loadmat('D:/ANT_3D_CNN/ANT_source_DL/control/ANT_con_source_3d_normalization.mat')
data2 = mat73.loadmat('D:/ANT_3D_CNN/ANT_source_DL/rbd/ANT_rbd_source_3d_normalization.mat')
con_1000 = data1['ANT_con_source_3d_normalization']
rbd_1000 = data2['ANT_rbd_source_3d_normalization']
con_label = np.zeros(shape=(np.size(con_1000, axis=0), 1))
rbd_label = np.ones(shape=(np.size(rbd_1000, axis=0), 1))
gt_con = data1['ground_truth']
gt_rbd = data2['ground_truth']
del data1, data2

# 62명 중 24명 random selection #
random_rbd_idx = np.random.choice(int(max(gt_rbd)), int(max(gt_con)), replace=False) + 1
random_rbd_idx = np.sort(random_rbd_idx)

# gt_rbd에서 뽑힌 24명 인덱스 확인 #
random_rbd = np.zeros(len(gt_rbd))
for rd_rbd_idx in random_rbd_idx:
    gt_random_rbd_idx = np.where(gt_rbd == rd_rbd_idx)
    random_rbd[gt_random_rbd_idx] = 1
del rd_rbd_idx

# 24명 데이터, gt 만들기 #
rbd_balance = np.squeeze(rbd_1000[np.where(random_rbd == 1), :, :, :])
rbd_balance_gt = gt_rbd[np.where(random_rbd == 1)]
rbd_balance_gt_final = np.zeros(len(rbd_balance_gt))
i = 1
for rd_rbd_idx in random_rbd_idx:
    final_idx = np.where(rbd_balance_gt == rd_rbd_idx)
    rbd_balance_gt_final[final_idx] = i
    i += 1
rbd_balance_gt_final_successive = rbd_balance_gt_final + 24

groups = np.concatenate((gt_con, rbd_balance_gt_final_successive), axis=0)
con_rbd = np.concatenate((con_1000, rbd_balance), axis=0)
con_rbd = np.nan_to_num(con_rbd)
con_rbd = np.reshape(con_rbd, (-1, 60, 120, 15, 1))
con_label = np.zeros(shape=(np.size(con_1000, axis=0), 1))
rbd_label = np.ones(shape=(np.size(rbd_balance, axis=0), 1))
con_rbd_label = np.concatenate((con_label, rbd_label), axis=0)
con_rbd_label_one_hot_encoding = np_utils.to_categorical(con_rbd_label)
del con_label, rbd_label, rbd_balance_gt_final_successive, rbd_balance_gt_final, rbd_balance_gt

# LOSOXV #
n_fold = 48         # 피험자 한 명씩 빼며 CV (LOSOXV)
cv = StratifiedGroupKFold(n_splits=n_fold)
i = 1
accuracy = []
train_accuracy = []
idx_order = []

for train_idxs, test_idxs in cv.split(con_rbd, con_rbd_label, groups):
    #print("TRAIN:", groups[train_idxs])
    print(" TEST 피험자 번호:", groups[test_idxs][0])
    model = Sequential()
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), input_shape=(60, 120, 15, 1), activation='relu'))
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

    history = model.fit(con_rbd[train_idxs], con_rbd_label_one_hot_encoding[train_idxs], epochs=20, batch_size=100, verbose=2, callbacks=[early_stopping_callback])

    k_accuracy = "%.8f" % (model.evaluate(con_rbd[test_idxs], con_rbd_label_one_hot_encoding[test_idxs])[1])
    #train_k_accuracy = "%.8f" % (model.evaluate(con_rbd[train_idxs], con_rbd_label_one_hot_encoding[train_idxs])[1])
    print("\n 현재 fold accuracy :", k_accuracy)
    accuracy.append(k_accuracy)
    #train_accuracy.append(train_k_accuracy)
    idx_order.append(max(groups[test_idxs]))

    # model.save('ANT_CNN_3D_con-rbd_%d.h5' % i)
    #model.save_weights('C:/Users/Nelab_001/PycharmProjects/pythonProject/ANT_CNN_3D_con-rbd_LOSOXV_model_weights/ANT_CNN_3D_con-rbd_%d.h5' % i)
    i += 1

    del model

print("\n %.8f fold accuracy : " % n_fold, accuracy)
np_acc = np.array(accuracy)
np_acc = np_acc.astype(np.double)
avg_acc = np.mean(np_acc)
std_acc = np.std(np_acc)
test_idx_order = np.array(idx_order)
#np_train_acc = np.array(train_accuracy)
print(" %.8f fold 평균 정확도 : " % n_fold, avg_acc)

terminate_time = timeit.default_timer()         # 종료 시간 체크
print("%f초 걸렸습니다." % (terminate_time - start_time))     # 딥러닝 돌아가는데 걸린 시간


"""notification to discord"""
# -------------------------------------------------------------------------------------------------------------------------------------------------
from knockknock import discord_sender
webhook_url = "https://discord.com/api/webhooks/1014081244734173274/kCGlk4rXRPb4LSOdf4ECz9P0lvbiHr5tq4EOR2Zf6RPd8OgU8eytgtG2-IjER1abFt4y"
@discord_sender(webhook_url=webhook_url)
def DL_notification():
    #import time
    #time.sleep(3)
    return {'averaged test acc' : avg_acc}, {'소요시간' :(terminate_time - start_time)} # Optional return value
DL_notification()
# -------------------------------------------------------------------------------------------------------------------------------------------------
