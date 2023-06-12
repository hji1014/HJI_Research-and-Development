# peak-amplification

from keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from keras.callbacks import EarlyStopping
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
from keras.utils import np_utils


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
            DA_d_p = DA_range.copy() * 1.1
            DA_d_m = DA_range.copy() * 0.9
            DA_d_p2 = DA_range.copy() * 1.2
            DA_d_m2 = DA_range.copy() * 0.8
            DA_d_p3 = DA_range.copy() * 1.3
            DA_d_m3 = DA_range.copy() * 0.7

            n[i, t-40:t+40] = DA_d_n
            p[i, t-40:t+40] = DA_d_p
            m[i, t-40:t+40] = DA_d_m
            p2[i, t-40:t+40] = DA_d_p2
            m2[i, t-40:t+40] = DA_d_m2
            p3[i, t-40:t+40] = DA_d_p3
            m3[i, t-40:t+40] = DA_d_m3

    # 3-times DA
    if aug_times == 3:
        x_train_aug = np.concatenate((n, p, m), axis=0)
        y_train_aug = np.concatenate((y_train, y_train, y_train), axis=0)

    # 5-times DA
    elif aug_times == 5:
        x_train_aug = np.concatenate((n, p, m, p2, m2), axis=0)
        y_train_aug = np.concatenate((y_train, y_train, y_train, y_train, y_train), axis=0)

    # 7-times DA
    else:
        x_train_aug = np.concatenate((n, p, m, p2, m2, p3, m3), axis=0)
        y_train_aug = np.concatenate((y_train, y_train, y_train, y_train, y_train, y_train, y_train), axis=0)

# model construction #

# model 1 - NDA
x = layers.Input(shape=(16800,))
h1 = layers.Activation('relu')(layers.Dense(300, name='zz')(x))
h2 = layers.Activation('relu')(layers.Dense(200)(h1))
h3 = layers.Activation('relu')(layers.Dense(100)(h2))
y = layers.Activation('softmax')(layers.Dense(2)(h3))
model = models.Model(x, y)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# model 2 - DA
x2 = layers.Input(shape=(16800,))
h1_2 = layers.Activation('relu')(layers.Dense(300, name='zz')(x2))
h2_2 = layers.Activation('relu')(layers.Dense(200)(h1_2))
h3_2 = layers.Activation('relu')(layers.Dense(100)(h2_2))
y2 = layers.Activation('softmax')(layers.Dense(2)(h3_2))
model2 = models.Model(x2, y2)
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.summary()

# train & test #
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping_callback])
model2.fit(x_train_aug, y_train_aug, epochs=100, batch_size=5, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping_callback])
print("\n Test Accuracy 1 : %.4f" % (model.evaluate(x_test, y_test, verbose=0)[1]))  # model.predict()로 test data 예측 결과 눈으로 볼 수 있음
print("\n Test loss 2 : %.4f" % (model.evaluate(x_test, y_test, verbose=0)[0]))
print("\n Test Accuracy 1 : %.4f" % (model2.evaluate(x_test, y_test, verbose=0)[1]))
print("\n Test loss 2 : %.4f" % (model2.evaluate(x_test, y_test, verbose=0)[0]))

# 중간에서 값 뽑기 #

#all_output=[layer.output for layer in model.layers]
x_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)

h1_dense = models.Model(inputs=model.input, outputs=model.layers[3].output)
h1_activation = models.Model(inputs=model.input, outputs=model.layers[4].output)
h2_dense = models.Model(inputs=model.input, outputs=model.layers[5].output)
h2_activation = models.Model(inputs=model.input, outputs=model.layers[6].output)

out1 = h1_dense.predict(x_tensor, steps=1)
out2 = h1_activation.predict(x_tensor, steps=1)
out3 = h2_dense.predict(x_tensor, steps=1)
out4 = h2_activation.predict(x_tensor, steps=1)

x_tensor2 = tf.convert_to_tensor(x_train_aug, dtype=tf.float32)

h1_dense2 = models.Model(inputs=model2.input, outputs=model2.layers[3].output)
h1_activation2 = models.Model(inputs=model2.input, outputs=model2.layers[4].output)
h2_dense2 = models.Model(inputs=model2.input, outputs=model2.layers[5].output)
h2_activation2 = models.Model(inputs=model2.input, outputs=model2.layers[6].output)

out5 = h1_dense2.predict(x_tensor2, steps=1)
out6 = h1_activation2.predict(x_tensor2, steps=1)
out7 = h2_dense2.predict(x_tensor2, steps=1)
out8 = h2_activation2.predict(x_tensor2, steps=1)


# TSNE

y_train_aug_tsne = np.where(y_train_aug == 0, 2, y_train_aug)
y_train_aug_tsne = np.where(y_train_aug_tsne == 1, 3, y_train_aug_tsne)
y_train_aug_tsne[:1200] = y_train

#out1_5 = np.concatenate((out1, out5), axis=0)
#out2_6 = np.concatenate((out2, out6), axis=0)
#out3_7 = np.concatenate((out3, out7), axis=0)
#out4_8 = np.concatenate((out4, out8), axis=0)

TSNE_result = TSNE(learning_rate=100, n_components=2)
TSNE_transformed = TSNE_result.fit_transform(out7)


#plt.figure(1)
#plt.rcParams["figure.figsize"] = (10, 10)
#plt.scatter(NDA_xs, NDA_ys, c=y_all)
#plt.show()

idx_0 = np.where(y_train_aug_tsne == 0)
idx_1 = np.where(y_train_aug_tsne == 1)
idx_2 = np.where(y_train_aug_tsne == 2)
idx_3 = np.where(y_train_aug_tsne == 3)

ori_con = TSNE_transformed[idx_0]
ori_rbd = TSNE_transformed[idx_1]
DA_con = TSNE_transformed[idx_2]
DA_rbd = TSNE_transformed[idx_3]

v = np.random.choice(len(DA_con), 400, replace=False)          # con or rbd 3000개 무작위로 뽑기
v_random = np.sort(v)
b = np.random.choice(len(DA_rbd), 400, replace=False)
b_random = np.sort(b)

plt.figure(1)
plt.rcParams["figure.figsize"] = (10, 10)
plt.scatter(ori_con[:, 0], ori_con[:, 1], c = plt.cm.seismic(0.4), label='original_control', s=60)
plt.scatter(ori_rbd[:, 0], ori_rbd[:, 1], c = plt.cm.seismic(0.6), label='original_RBD', s=60)
plt.scatter(DA_con[v_random, 0], DA_con[v_random, 1], c = plt.cm.seismic(0.1), label='augmented_control', s=20)
plt.scatter(DA_rbd[b_random, 0], DA_rbd[b_random, 1], c = plt.cm.seismic(0.9), label='augmented_RBD', s=20)
plt.legend()
plt.show()

plt.figure(2)
plt.rcParams["figure.figsize"] = (10, 10)
plt.scatter(DA_con[:, 0], DA_con[:, 1], c = plt.cm.seismic(0.6), label='augmented_control')
plt.scatter(ori_con[:, 0], ori_con[:, 1], c = plt.cm.seismic(0.4), label='original_control')
plt.legend()
plt.show()

plt.figure(3)
plt.rcParams["figure.figsize"] = (10, 10)
plt.scatter(ori_con[:, 0], ori_con[:, 1], c = plt.cm.seismic(0.4), label='original_control')
plt.legend()
plt.show()
