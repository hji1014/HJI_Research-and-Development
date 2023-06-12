from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization
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
from sklearn.metrics import confusion_matrix, roc_curve, auc

confusion_all = np.array([[0, 0], [0, 0]])
tprs = np.zeros((62, 100))
#fprs = []
aucs = np.zeros((62, 1))

for i in range(62):

    sub = i + 1

    x_test = np.load('D:/ANT_3D_CNN_source_transfer2/premodel_testset/premodel_x_sub_%i.npy' % sub)
    y_test = np.load('D:/ANT_3D_CNN_source_transfer2/premodel_testset/premodel_y_sub_%i.npy' % sub)

    #model = load_model('D:/ANT_3D_CNN_source_transfer2/transfer_model/transfer_model_sub_%i.h5' % sub)

    model = Sequential()
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), kernel_initializer='he_normal', input_shape=(60, 120, 15, 1), activation='relu'))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(0.25))  # turn off of 25% nodes
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.load_weights('D:/ANT_3D_CNN_source_transfer2/premodel_weights/premodel_weights_sub_%i.h5' % sub)

    predict = model.predict(x_test)
    y_predict = np.where(predict >= 0.5, 1, 0)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    cf_matrix = np.array([[tn, fp], [fn, tp]])
    confusion_all = confusion_all + cf_matrix

    # AUROC curve
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(y_test, predict, pos_label=1)
    roc_auc = auc(fpr, tpr)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    tprs[i, :] = interp_tpr
    aucs[i, 0] = roc_auc


fig, ax = plt.subplots()

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic curve")
plt.legend(loc="lower right")
plt.show()



############################################################ cf matrix 시각화 ###################################
import seaborn as sns

cf_matrix = np.sum(confusion_all, axis=0)

group_names = ['True Negative','False Positive','False Negative','True Positive']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     np.divide(cf_matrix, np.sum(cf_matrix, axis=0)).flatten()]
categories = ['NC','iRBD']

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

cf_matrix2 = np.divide(cf_matrix, np.sum(cf_matrix, axis=0))*100
fig, ax = plt.subplots(figsize=(6, 4))         # Sample figsize in inches
sns.heatmap(cf_matrix2, annot=labels, fmt='',xticklabels=categories, yticklabels=categories, cmap='Blues', ax=ax, annot_kws={"size": 16})
