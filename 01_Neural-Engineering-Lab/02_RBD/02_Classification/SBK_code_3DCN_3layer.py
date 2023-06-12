#CNN LRP
#tensorflow=3.6
#spyder 3.3.6
##https://github.com/albermax/innvestigate/blob/master/examples/notebooks/mnist_compare_methods.ipynb

savepath = 'E:\yonsei\\ANT_CNN_new\\ready-made\\'

###############################################################################
###############################################################################
###############################################################################
import warnings
warnings.simplefilter('ignore')

import numpy as np

import keras
import keras.backend
import keras.models
#import keras.backend.tensorflow_backend as K

from keras.optimizers import Adam

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils
import innvestigate.utils.tests
import innvestigate.utils.tests.networks

import tensorflow as tf
###############################################################################
###############################################################################
###############################################################################
def normalize(X, input_range=[0, 1]):
    """
    Generically shifts data from interval [a, b] to interval [c, d].
    Assumes that theoretical min and max values are populated.
    """

    if len(input_range) != 2:
        raise ValueError(
            "Input range must be of length 2, but was {}".format(
                len(input_range)))
    if input_range[0] >= input_range[1]:
        raise ValueError(
            "Values in input_range must be ascending. It is {}".format(
                input_range))

    a, b = X.min(), X.max()
    c, d = input_range

    def preprocessing(X):
        # shift original data to [0, b-a] (and copy)
        X = X - a
        # scale to new range gap [0, d-c]
        X /= (b-a)
        X *= (d-c)
        # shift to desired output range
        X += c
        return X

    def revert_preprocessing(X):
        X = X - c
        X /= (d-c)
        X *= (b-a)
        X += a.
        return X

    return preprocessing, revert_preprocessing
###############################################################################
###############################################################################
###############################################################################
def postprocess(X):
    X = X.copy()
    X = iutils.postprocess_images(X)
    return X
###############################################################################
###############################################################################
###############################################################################
# Load data
# returns x_train, y_train, x_test, y_test as numpy.ndarray
# returns x_train, y_train, x_test, y_test as numpy.ndarray
import scipy.io as scio
import hdf5storage
mat = hdf5storage.loadmat(savepath+'input_all_100ms.mat');
original_X = mat['input']
original_y = mat['output']
original_y = np.reshape(original_y,[len(original_y),])
del mat
###############################################################################
###############################################################################
###############################################################################
##CON-RBD 숫자맞추기
#ind_0 = np.where(original_y == 0) # CONTROL
#ind_1 = np.where(original_y != 0) # RBD (MCI=1, nMCI=2)
#ind_vec = [ind_0[0],ind_1[0]]

#smaller_class = np.argmin([len(ind_0[0]), len(ind_1[0])])
#larger_class = np.argmax([len(ind_0[0]), len(ind_1[0])])

#random_indices = np.random.permutation(len(ind_vec[smaller_class]))
#ind_to_use = np.concatenate((ind_vec[larger_class][random_indices], ind_vec[smaller_class]))
#ind_to_use = ind_to_use[np.random.permutation(len(ind_to_use))]
total_x = original_X#[ind_to_use,:,:,:]
total_y = original_y#[ind_to_use,]
#total_x = np.reshape(total_x,[len(ind_to_use),60,120,10,1]) #3D
total_x = np.reshape(total_x,[len(original_X),60,120,10,1]) #3D

#ind_2 = np.where(total_y != 0)
#total_y[ind_2[0]] = 1
###############################################################################
#for test_num in range(9):
#test_num = 0

# Create preprocessing functions
input_range = [-1, 1]
preprocess, revert_preprocessing = normalize(total_x, input_range)
# Preprocess data
data_total_x = preprocess(total_x)
data_total_y = total_y

num_classes = len(np.unique(data_total_y))

del total_x,total_y
del original_y, original_X
###############################################################################
###############################################################################
###############################################################################
#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

validation_split = 0
num_folds = 10
# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
# Merge inputs and targets

# Define the K-fold Cross Validator
#kfold = KFold(n_splits=num_folds, shuffle=True)
skf = StratifiedKFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
#for train, test in kfold.split(data_total_x, data_total_y):
for train, test in skf.split(data_total_x, data_total_y):    
        # Create & train model
    input_shape = (60, 120, 10, 1)
    model = keras.models.Sequential([
        keras.layers.BatchNormalization(input_shape=input_shape),    
        keras.layers.Conv3D(10, (3, 3, 2), activation="relu", input_shape=input_shape),     
        keras.layers.Conv3D(10, (3, 3, 2), activation="relu"),
        keras.layers.MaxPooling3D((3, 3, 2)),
        keras.layers.Dropout(0.2),
        #keras.layers.BatchNormalization(),
        keras.layers.Conv3D(20, (2, 2, 2), activation="relu"),
        keras.layers.Conv3D(20, (2, 2, 2), activation="relu"),
        keras.layers.MaxPooling3D((2, 2, 2)),
        keras.layers.Dropout(0.2),
        #keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(200, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    #model.summary()
    # convert class vectors to binary class matrices
    batch_size=100
    epochs=30
    #with K.tf.device('/gpu:0'):
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
      # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    #with K.tf.device('/gpu:0'):
    train_history = model.fit(data_total_x[train], keras.utils.to_categorical(data_total_y[train], num_classes),
                    batch_size=batch_size, validation_split=validation_split, epochs=epochs, verbose=1)
    
    scores = model.evaluate(data_total_x[test],keras.utils.to_categorical(data_total_y[test], num_classes), verbose=0)
    
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    test_output = model.predict_classes(data_total_x[test])
    test_answer = data_total_y[test]
    
    ###############################################################################
    scio.savemat(savepath+'3D3_100\\'+'fold_no'+str(fold_no)+'_accuracy3D.mat',{'history': train_history.history, 'loss': scores[0], 'accuracy_test':scores[1]})
    scio.savemat(savepath+'3D3_100\\'+'fold_no'+str(fold_no)+'_predictions3D.mat',{'train':train, 'test': test, 'target': test_answer, 'outputs':test_output})
    model.save(savepath+'3D3_100\\'+'fold_no'+str(fold_no)+'model3D.h5')
    fold_no = fold_no + 1










    ###############################################################################
    # Create model without trailing softmax  
savepath = 'E:\yonsei\\ANT_CNN_new\\ready-made\\'
import scipy.io as scio
import hdf5storage
from keras.models import load_model

import numpy as np

import keras
import keras.backend
import keras.models
#import keras.backend.tensorflow_backend as K

from keras.optimizers import Adam

import innvestigate
import innvestigate.utils as iutils

import innvestigate.utils
import innvestigate.utils.tests
import innvestigate.utils.tests.networks


mat = hdf5storage.loadmat(savepath+'input_all_100ms.mat');
original_X = mat['input']
original_y = mat['output']
original_y = np.reshape(original_y,[len(original_y),])
del mat

total_x = original_X
total_y = original_y
total_x = np.reshape(total_x,[len(original_X),60,120,10,1]) #3D
input_range = [-1, 1]
preprocess, revert_preprocessing = normalize(total_x, input_range)
# Preprocess data
data_total_x = preprocess(total_x)
data_total_y = total_y

num_classes = len(np.unique(data_total_y))
num_folds = 10

del total_x,total_y
del original_y, original_X

fold_no = 1
while fold_no <= num_folds:
    mat = scio.loadmat(savepath+'3D3_100\\'+'fold_no'+str(fold_no)+'_predictions3D.mat')
    model = load_model(savepath+'3D3_100\\'+'fold_no'+str(fold_no)+'model3D.h5')
    test = mat['test']
    test = np.reshape(test,[len(test[0]),])
    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    model_wo_softmax.summary()
    analyzer = innvestigate.create_analyzer("deep_taylor.bounded",        # analysis method identifier
                                        model_wo_softmax, # model without softmax output
                                        **{"low": input_range[0],"high": input_range[1]})      # optional analysis parameters
    LRPs = []
    for LRPloop in range(len(test)):
        print(round(LRPloop/len(test)*100))
        image = data_total_x[test][LRPloop:LRPloop+1]
        #DeepTaylor LRP
        LRP = analyzer.analyze(image)
        LRP = postprocess(LRP)
        LRPs.append(LRP.squeeze())
    scio.savemat(savepath+'3D3_100\\'+'fold_no'+str(fold_no)+'LRPresult3D.mat',{'LRPanalysis':LRPs})
      # Increase fold number
    del LRPs,LRP,model
    fold_no = fold_no + 1