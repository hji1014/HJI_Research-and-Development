#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv("./dataset/housing.csv", delim_whitespace=True, header=None)
'''
print(df.info())
print(df.head())
'''
dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer='adam')

model.fit(X_train, Y_train, epochs=200, batch_size=10)

# 예측 값과 실제 값의 비교
Y_prediction = model.predict(X_test).flatten()
pred = []
for i in range(152):
    label = Y_test[i]
    prediction = Y_prediction[i]
    #print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))
    pred.append(prediction)

import numpy as np
mse = mean_squared_error(Y_test, pred)
rmse = np.sqrt(mse)
r2_points = r2_score(Y_test, pred)
print("RMSE:", rmse)
print("R^2:", r2_points)

model.save_weights('model_weights.h5')