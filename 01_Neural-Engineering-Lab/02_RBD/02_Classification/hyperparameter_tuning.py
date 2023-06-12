"""https://webnautes.tistory.com/1642"""
"""https://www.analyticsvidhya.com/blog/2021/06/create-convolutional-neural-network-model-and-optimize-using-keras-tuner-deep-learning/"""

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization
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
import keras_tuner

from tensorflow.python.keras.callbacks import TensorBoard
from time import time

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

con_rbd = np.concatenate((con_1000, rbd_1000), axis=0)
con_rbd = np.nan_to_num(con_rbd)
con_rbd_label = np.concatenate((con_label, rbd_label), axis=0)
del con_1000, rbd_1000, con_label, rbd_label

# train/validation #
sf = np.random.choice(int(np.size(con_rbd, axis=0)), int(np.size(con_rbd, axis=0)), replace=False)
zz = np.reshape(con_rbd, (-1, 60, 120, 15, 1))
zzz = con_rbd_label
zz = zz[sf, :, :, :, :]
zzz = zzz[sf, :]
zzzz = np_utils.to_categorical(zzz)
del con_rbd, con_rbd_label

X_train, X_test, Y_train, Y_test = train_test_split(zz, zzzz, test_size=0.1, random_state=seed)
del zz, zzz, zzzz

# activation function 종류 : ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
# loss function 종류 : ['binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'poisson']
# optimizer 종류 : ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']

def build_model(hp):

    model_activation = hp.Choice('best_model_activation', values=['relu', 'selu', 'elu'])
    model_dropout = hp.Float('best_model_dropout', min_value=0.1, max_value=0.8, step=0.1)
    model_dense_node = hp.Int('best_model_dense_node', min_value=64, max_value=256, step=64)

    model = Sequential()
    model.add(Conv3D(
        filters=hp.Int('conv_1_filter', min_value=16, max_value=48, step=16),
        kernel_size=hp.Int('conv_1_kernel', min_value=2, max_value=3, step=1),
        input_shape=(60, 120, 15, 1),
        activation=model_activation,
        padding='same'))
    model.add(Conv3D(
        filters=hp.Int('conv_2_filter', min_value=16, max_value=48, step=16),
        kernel_size=hp.Int('conv_2_kernel', min_value=2, max_value=3, step=1),
        activation=model_activation,
        padding='same'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(model_dropout))  # turn off of 25% nodes
    model.add(Conv3D(
        filters=hp.Int('conv_3_filter', min_value=16, max_value=48, step=16),
        kernel_size=hp.Int('conv_3_kernel', min_value=2, max_value=3, step=1),
        activation=model_activation,
        padding='same'))
    model.add(Conv3D(
        filters=hp.Int('conv_4_filter', min_value=16, max_value=48, step=16),
        kernel_size=hp.Int('conv_4_kernel', min_value=2, max_value=3, step=1),
        activation=model_activation,
        padding='same'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(model_dropout))
    model.add(Flatten())
    model.add(Dense(
        model_dense_node,
        activation=model_activation))
    model.add(Dropout(model_dropout))
    model.add(Dense(
        model_dense_node,
        activation=model_activation))
    model.add(Dropout(model_dropout))
    model.add(Dense(
        2,
        activation='softmax'))
    Adam = optimizers.Adam(hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001]))
    model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
    model.summary()

    return model

# tuning 시작
from kerastuner import RandomSearch                                                 # importing random search
tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=20)          # creating randomsearch object
tuner.search(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test))         # search best parameter

# 소요시간 측정
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time))


"""notification to discord"""
# -------------------------------------------------------------------------------------------------------------------------------------------------
from knockknock import discord_sender
webhook_url = "https://discord.com/api/webhooks/1014081244734173274/kCGlk4rXRPb4LSOdf4ECz9P0lvbiHr5tq4EOR2Zf6RPd8OgU8eytgtG2-IjER1abFt4y"
@discord_sender(webhook_url=webhook_url)
def DL_notification():
    #import time
    #time.sleep(3)
    return {'소요시간' :(terminate_time - start_time)} # Optional return value
DL_notification()
# -------------------------------------------------------------------------------------------------------------------------------------------------
