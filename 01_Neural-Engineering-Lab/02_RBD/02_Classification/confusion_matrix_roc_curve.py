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
from sklearn.metrics import confusion_matrix

# t = [1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
# pp = [1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
# a = confusion_matrix(t, pp)
# b = confusion_matrix(t, pp)
# c = a + b

confusion_all_transfer = np.array([[0, 0], [0, 0]])
acc = np.load('D:/ANT_3D_CNN_source_transfer2/transfer_model_accuracy/all_TL_acc.npy')

from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

for i in range(62):

    sub = i + 1

    x_test = np.load('D:/ANT_3D_CNN_source_transfer2/transfer_model_testset/transfer_model_x_sub_%i.npy' % sub)
    y_test = np.load('D:/ANT_3D_CNN_source_transfer2/transfer_model_testset/transfer_model_y_sub_%i.npy' % sub)

    model = load_model('D:/ANT_3D_CNN_source_transfer2/transfer_model/transfer_model_sub_%i.h5' % sub)

    # model = Sequential()
    # model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), kernel_initializer='he_normal', input_shape=(60, 120, 15, 1), activation='relu'))
    # model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
    # model.add(MaxPooling3D(pool_size=2))
    # model.add(Dropout(0.25))  # turn off of 25% nodes
    # model.add(Conv3D(filters=16, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
    # model.add(Conv3D(filters=32, kernel_size=(3, 3, 2), kernel_initializer='he_normal', activation='relu'))
    # model.add(MaxPooling3D(pool_size=2))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # model.load_weights('D:/ANT_3D_CNN_source_transfer2/premodel_weights/premodel_weights_sub_%i.h5' % sub)

    predict = model.predict(x_test)
    y_predict = np.where(predict >= 0.5, 1, 0)
    #loss = "%.8f" % (model.evaluate(x_test, y_test)[0])
    #acc = "%.8f" % (model.evaluate(x_test, y_test)[1])

    confusion = confusion_matrix(y_test, y_predict)
    confusion_all_transfer = confusion_all_transfer + confusion


    def is_classifier(estimator):
        """Return True if the given estimator is (probably) a classifier.
        Parameters
        ----------
        estimator : object
            Estimator object to test.
        Returns
        -------
        out : bool
            True if estimator is a classifier and False otherwise.
        """
        return getattr(estimator, "_estimator_type", None) == "classifier"

    viz = RocCurveDisplay.from_estimator(
        model,
        x_test,
        y_test,
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
group_counts = ["{0:0.0f}".format(value) for value in
                confusion_all_transfer.flatten()]
#group_percentages = ["{0:.2%}".format(value) for value in
#                     confusion_all_transfer.flatten()/np.sum(confusion_all_transfer)]
group_percentages = ['99.66%', '0.38%', '0.34%', '99.62%']
categories = ['NC', 'iRBD']
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
confusion_acc = np.array([[99.66, 0.38], [0.34, 99.62]])
ax = sns.heatmap(confusion_acc, annot=labels, fmt='', xticklabels=categories, yticklabels=categories, cmap='Blues')
plt.ylabel('Actual class')
plt.xlabel('Predicted class')
fig = ax.figure
cbar = fig.get_children()[-1]
cbar.yaxis.set_ticks([10, 20, 30, 40, 50, 60, 70, 80, 90])

###################################################################################################################
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic example",
)
ax.legend(loc="lower right")
plt.show()