#! /usr/bin/env python
#coding=utf-8
#! /usr/bin/env python
#coding=utf-8
from __future__ import print_function
from __future__ import division
from functools import reduce
import re
import tarfile
import math

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, RepeatVector, Activation, Flatten
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D

from sklearn.metrics import average_precision_score

from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence

RNN = recurrent.LSTM

EMBED_SIZE = 32
HIDDEN_SIZE= 16
MAX_LEN= 200
BATCH_SIZE = 16
EPOCHS = 5 

def readResult(y_test,results):
    index=0
    p=n=tp=tn=fp=fn=0
    for prob in results:
        if prob>0.5:
            predLabel=1
        else:
            predLabel=0
        if y_test[index]>0:
            p+=1
            if predLabel>0:
                tp+=1
            else:
                fn+=1
        else:
            n+=1
            if predLabel==0:
                tn+=1
            else:
                fp+=1
        index+=1

    acc=(tp+tn)/(p+n)
    precisionP=tp/(tp+fp)
    precisionN=tn/(tn+fn)
    recallP=tp/(tp+fn)
    recallN=tn/(tn+fp)
    gmean=math.sqrt(recallP*recallN)
    f_p=2*precisionP*recallP/(precisionP+recallP)
    f_n=2*precisionN*recallN/(precisionN+recallN)
    print ('{gmean:%s recallP:%s recallN:%s} {precP:%s precN:%s fP:%s fN:%s} acc:%s' %(gmean,recallP,recallN,precisionP,precisionN,f_p,f_n,acc))
    print('AUC %s' %average_precision_score(y_test,results))

    output=open('result.output','w')
    output.write('\n'.join(['%s' %r for r in results]))

def lstm_prediction(X_train,y_train,X_test,y_test,vocab_size): 
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape) 
    
    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocab_size, EMBED_SIZE, input_length=MAX_LEN, dropout=0.2))
    
    
    model.add(RNN(HIDDEN_SIZE))

    model.add(Dense(1, activation='sigmoid'))

#    model.compile(optimizer='adam',
#                 loss='binary_crossentropy')
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE,
              nb_epoch=EPOCHS, show_accuracy=True,
              validation_data=(X_test, y_test))
            
    X_pred=model.predict(X_test)
    results=[result[0] for result in X_pred]
    
    return readResult(y_test,results)

def lstm_train(X_train,y_train,vocab_size): 
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)

#    print('X_train shape:', X_train.shape)
#    print('X_test shape:', X_test.shape) 
    
    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocab_size, EMBED_SIZE, input_length=MAX_LEN, dropout=0.2))
    
    model.add(RNN(HIDDEN_SIZE))

    model.add(Dense(1, activation='sigmoid'))

 #   model.compile(optimizer='adam',
 #                 loss='binary_crossentropy')
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE,
              nb_epoch=EPOCHS, show_accuracy=True)
    
    return model
