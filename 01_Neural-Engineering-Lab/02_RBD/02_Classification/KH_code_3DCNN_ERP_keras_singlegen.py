#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:49:08 2021

@author: nelab

sequence classification [binary_crossentropy]

https://www.tensorflow.org/guide/keras/rnn?hl=ko
https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7

"""

import os
os.chdir('/media/nelab/Hyun/posner/초기데이터_pipeline/ft_pipeline')

import numpy as np
import scipy.io as scio
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf

# Datasets
ABS_Path = '/media/nelab/Hyun/posner/초기데이터_pipeline/ft_pipeline/Analysis_EEG/dl_input_dataset/'

annot_file_path = ABS_Path + '/ERP_source_whole_trials_12/mark.mat'
root_path = ABS_Path + 'ERP_source_whole_trials_12'


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, annot_file, root_path, batch_size, dim,
                 n_classes, shuffle=True, list_idx=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.root_dir = root_path
        self.marks = scio.loadmat(annot_file)['R_G_all']
#        self.labels = labels
#        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indices = list_idx
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [(k, os.path.join(self.root_dir, self.marks[k, 0][0])) for k in indexes]

        # Generate data
        data_, labels_ = self.__data_generation(list_IDs_temp)
        # sample = {'data': data, 'label': labels}
        data = tf.convert_to_tensor(data_)
        labels = tf.convert_to_tensor(labels_)
        # labels = tf.keras.utils.to_categorical(y, num_classes=2)

        # return sample
        return (data, labels)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.indices is None:
            self.indexes = np.arange(len(self.marks))
        else:
            self.indexes = self.indices
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        # X = np.empty((self.batch_size, self.dim))
        y = np.empty((self.batch_size), dtype='float')

        # Generate data
        for i, (k, ID) in enumerate(list_IDs_temp):
            
            # Store sample
            X[i,] = scio.loadmat(ID)['X_all'][:,:,:,np.newaxis] # time_X_Y_channel
            
            # X[i,] = scio.loadmat(ID)['X_all'].reshape(-1) # for MLP test
            # Store class
            y[i,] = np.array(self.marks[k,1][0] == 2,dtype='float') # 0: CON, 1: RBD (annots: 1: CON, 2: RBD)
            
        return X, y


# Create model
if keras.backend.image_data_format == "channels_first":
    input_shape = (1, 12, 120, 120)
else:
#    input_shape = (60, 120, 1)
    input_shape = (12, 120, 120, 1)
    
# Design model 
# input_shape = (12*120*120)

def MLP_keras():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape = input_shape),
        keras.layers.Dense(500, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
        ])
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
    
    return model


def CNN1D_keras():
    model = keras.models.Sequential([
        # keras.layers.Flatten(input_shape = input_shape),
        keras.layers.Reshape((-1,1), input_shape=input_shape),
        keras.layers.Conv1D(6, 3, activation="relu"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(16, 3, activation="relu"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(16, 3),
        keras.layers.BatchNormalization(),
        keras.layers.Activation(activation="relu"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation="relu"),
        keras.layers.Dense(84, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
        ])
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
    
    return model


def CNN3D_keras():
    model = keras.models.Sequential([
        keras.layers.Conv2D(6, (3, 3, 3), activation="relu", input_shape=input_shape),
        keras.layers.MaxPooling3D((2,2,2)),
        keras.layers.Conv2D(16, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(16, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.summary()
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])



# create model: CNN3DClassifier_BN_LRP_keras

def CNN3DClassifier_BN_LRP_keras():
    CNN3DClassifier_BN_LRP_keras = keras.models.Sequential([ # conv 이후 batch norm. overfitting 발생안함. 'CNN3DClassifier_BN_LRP_keras'
        keras.layers.Conv3D(6, (3, 3, 3), activation="relu", input_shape=input_shape),
        keras.layers.MaxPooling3D((2, 2, 2)),
    #            keras.layers.Conv3D(16, (3, 3, 3), activation="relu"),
        keras.layers.Conv3D(16, (3, 3, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation(activation="relu"),
        keras.layers.MaxPooling3D((2, 2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
    #            keras.layers.Dropout(0.15),
        keras.layers.Dense(num_classes, activation="sigmoid"),
    ])
    CNN3DClassifier_BN_LRP_keras.summary()
    CNN3DClassifier_BN_LRP_keras.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])

# tf.keras.backend.clear_session()

CNN = keras.models.load_model('Analysis_EEG/dl_output_ERP_source/CNN3DClassifier_BN_LRP_keras_batch_256/CNN3DClassifier_BN_LRP_keras_1fold_1try.h5')
model = keras.Model(inputs=CNN.input, outputs=CNN.get_layer('dense_20').output)
# create model: CNN_LSTM_LRP_keras

def CNN_LSTM_LRP_keras(): 
    model = keras.models.Sequential([ # conv 이후 batch norm. overfitting 발생안함
        keras.layers.TimeDistributed(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape)),
        keras.layers.TimeDistributed(keras.layers.MaxPool2D((2,2))),
        keras.layers.TimeDistributed(keras.layers.Conv2D(32, (3, 3), activation="relu")),
        keras.layers.TimeDistributed(keras.layers.MaxPool2D((2,2))),
        keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), activation="relu")),
        keras.layers.TimeDistributed(keras.layers.MaxPool2D((2,2))),
        keras.layers.TimeDistributed(keras.layers.Dropout(0.3)),
        keras.layers.TimeDistributed(keras.layers.Flatten()),
        keras.layers.TimeDistributed(keras.layers.Dense(64, activation="relu")),
        keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True), merge_mode='concat'),
        keras.layers.Dropout(0.2),
        keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True), merge_mode='concat'),
        keras.layers.Dropout(0.2),
        keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True), merge_mode='concat'),
        keras.layers.Dropout(0.2),
        keras.layers.Bidirectional(keras.layers.LSTM(32), merge_mode='concat'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.build((None, 12, 120, 120, 1))
    model.summary()
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
    return model



def CNN_LSTM_LRP_keras_failed(): 
    # model = keras.models.Sequential([ # conv 이후 batch norm. overfitting 발생안함
    #     keras.layers.TimeDistributed(keras.layers.Conv2D(6, (3, 3), activation="relu", input_shape=input_shape)),
    #     keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2))),
    #     keras.layers.TimeDistributed(keras.layers.Conv2D(16, (3, 3))),
    #     keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
    #     keras.layers.TimeDistributed(keras.layers.Activation(activation="relu")),
    #     keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2))),
    #     keras.layers.TimeDistributed(keras.layers.Flatten()),
    #     keras.layers.Bidirectional(keras.layers.LSTM(100), merge_mode='concat'),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(100, activation="relu"),
    #     keras.layers.Dense(1, activation="sigmoid"),
    # ])
    model = keras.models.Sequential([ # conv 이후 batch norm. overfitting 발생안함
        keras.layers.TimeDistributed(keras.layers.Conv2D(6, (3, 3), activation="relu", input_shape=input_shape)),
        keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2))),
        keras.layers.TimeDistributed(keras.layers.Conv2D(16, (3, 3), activation="relu")),
        keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2))),
        keras.layers.TimeDistributed(keras.layers.Conv2D(16, (3, 3), activation="relu")),
        keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2))),
        keras.layers.TimeDistributed(keras.layers.Flatten()),
        keras.layers.Bidirectional(keras.layers.LSTM(100), merge_mode='concat'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    # model.build((None, 12, 120, 120, 1))
    # model.summary()
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
    return model

# create model: 

def CNN_LSTM_LRP_keras_simple():
    model = keras.models.Sequential([
        keras.layers.TimeDistributed(keras.layers.Conv2D(6, (3, 3), activation="relu", input_shape=input_shape)),
        keras.layers.TimeDistributed(keras.layers.MaxPool2D((2,2))),
        keras.layers.TimeDistributed(keras.layers.Flatten()),
        # keras.layers.Bidirectional(keras.layers.LSTM(100,  return_sequences=True), merge_mode='concat'),
        # keras.layers.Dropout(0.5),
        keras.layers.Bidirectional(keras.layers.LSTM(100), merge_mode='concat'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    # model.build((None, 12, 120, 120, 1))
    # model.summary()
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
    return model

# create model: LSTM_LRP_keras

def LSTM_keras():
    model = keras.models.Sequential([ # conv 이후 batch norm. overfitting 발생안함. 'CNN3DClassifier_BN_LRP_keras'
        keras.layers.TimeDistributed(keras.layers.Flatten(), input_shape = input_shape),
        # keras.layers.Bidirectional(keras.layers.LSTM(100,  return_sequences=True), merge_mode='concat'),
        # keras.layers.Dropout(0.5),
        keras.layers.Bidirectional(keras.layers.LSTM(100), merge_mode='concat'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    # CNN_LSTM_LRP_keras.build((None, 12, 120, 120, 1))
    # model.summary()
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
    return model

def LSTM_keras_uni():
    model = keras.models.Sequential([ # conv 이후 batch norm. overfitting 발생안함. 'CNN3DClassifier_BN_LRP_keras'
        keras.layers.TimeDistributed(keras.layers.Flatten(), input_shape = input_shape),
        # keras.layers.Bidirectional(keras.layers.LSTM(100,  return_sequences=True), merge_mode='concat'),
        # keras.layers.Dropout(0.5),
        keras.layers.LSTM(100),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    # LSTM_keras_uni.build((None, 12, 120, 120, 1))
    # model.summary()
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
    return model


# create model: 

def ConvLSTM_keras():
    model = keras.models.Sequential([ # conv 이후 batch norm. overfitting 발생안함
        keras.layers.ConvLSTM2D(64, (3,3), input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    # model.build((None, 12, 120, 120, 1))
    # model.summary()
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
    return model

# Hyper-parameters 
#input_size = dimension
num_classes = 1
num_epochs = 100
learning_rate = 0.001

# set variables
repetition = 1 # CV 반복 횟수
seed = 7 # random seed
n_split = 10 # train/valid/test = 8:1:1
val_split_ratio = 1/9 # split ratio for validation set

# Parameters
params = {'dim': input_shape,
          'batch_size': 256,
          'n_classes': 1,
          'shuffle': True,
          'list_idx': None}

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

marks = scio.loadmat(annot_file_path)['R_G_all']
train_acc_all = []
test_acc_all = []
es = keras.callbacks.EarlyStopping(monitor='accuracy', mode = 'max', patience=8)

for ri in range(0, repetition):
    cv = KFold(n_split, shuffle=True, random_state=seed)
    for ci, (idx_train, idx_test) in enumerate(cv.split(marks)):
        # if ci == 1:
        #     ci -= 1
        #     break
        idx = {}
        idx['train'] = idx_train
        idx['test'] = idx_test
        print('\nnow training: '+str(ci+1)+' / '+str(10)+' from '+str(ri+1)+' of '+str(repetition))
#        -----------------------------------------------------------------------      
        model_name = 'LSTM_keras_uni()'
        model = eval(model_name)
#        -----------------------------------------------------------------------
        train_idx, val_idx = train_test_split(idx['train'], random_state=66, test_size=val_split_ratio)
        
        
        params['list_idx'] = train_idx
        trn_loader = DataGenerator(annot_file=annot_file_path, root_path=root_path, **params)
        params['list_idx'] = val_idx
        val_loader = DataGenerator(annot_file=annot_file_path, root_path=root_path, **params)
        params['list_idx'] = idx_test
        params['shuffle'] = False
        test_loader = DataGenerator(annot_file=annot_file_path, root_path=root_path, **params)
        
        total_step = len(trn_loader)
        
        # Train model on dataset
        hstry = model.fit_generator(generator=trn_loader, epochs=num_epochs, validation_data=val_loader,
                            use_multiprocessing=False, callbacks=[es])
        # model.fit_generator(generator=trn_loader, epochs=num_epochs, validation_data=val_loader,
        #                     use_multiprocessing=True, workers=4)
        # model.fit_generator(generator=trn_loader,
        #                     use_multiprocessing=True,
        #                     workers=6)
        
        scores = model.evaluate_generator(test_loader)
        print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))
        
        train_acc_all.append(hstry.history['accuracy'][-1])
        test_acc_all.append(scores[-1])
        
        preds = model.predict_generator(test_loader)>0.5 # batch_size*steps

        # Save the model checkpoint
        model.save('Analysis_EEG/dl_output_ERP_source/'+model_name+'_batch_256'+'/'+model_name+'_'+str(ci+1)+'fold_'+str(ri+1)+'try.h5')
        # Save result
        np.save('Analysis_EEG/dl_output_ERP_source/'+model_name+'_batch_256'+'/'+model_name+'_idx_'+str(ci+1)+'fold_'+str(ri+1)+'try.npy', idx, allow_pickle = True)
        

# evaluation
print('train acc: ' + str(round(np.array(train_acc_all).mean()*100, ndigits=2)) 
+ ' ± ' + str(round(np.array(train_acc_all).std()*100, ndigits=2)) + ' %')
print('test acc: ' + str(round(np.array(test_acc_all).mean()*100, ndigits=2)) 
+ ' ± ' + str(round(np.array(test_acc_all).std()*100, ndigits=2)) + ' %')
            
np.savez('Analysis_EEG/dl_output_ERP_source/'+model_name+'_batch_256'+'/'+model_name+'_result.npz', tr=train_acc_all, te=test_acc_all)


""" LRP """
ci = 0
ri = 0
model_name = 'LSTM_LRP_keras()'
model = eval(model_name)
test = keras.models.load_model('Analysis_EEG/dl_output_ERP_source/'+model_name+'_batch_256'+'/'+model_name+'_'+str(ci+1)+'fold_'+str(ri+1)+'try.h5')







"""dummy codes"""

# 'Train Model'
# import numpy as np
# import tensorflow.keras as keras

# Generators
training_generator = DataGenerator(annot_file=annot_file_path, root_dir=root_path, **params)
params['list_idx'] = idx['validation']
validation_generator = DataGenerator(annot_file=annot_file_path, root_dir=root_path, **params)

# Design model
#model = keras.Sequential(
#    [
#        layers.Dense(500, activation="relu", name="layer1"),
#        layers.Dense(100, activation="relu", name="layer2"),
#        layers.Dense(1, name="layer3"),
#    ]
#)
#model.compile()

model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    use_multiprocessing=True,
                    workers=6)




## Dummy codes::
    
    
# Design model - MLP for test
input_shape = (12*120*120)
MLP_test = keras.Sequential(
    [
        layers.Dense(500, activation="relu", input_shape=(12*120*120,), name="layer1"),
        layers.Dense(100, activation="relu", name="layer2"),
        layers.Dense(1, activation="sigmoid", name="layer3"),
    ]
)
MLP_test.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])

# MLP_test.fit(dataset.batch(16))