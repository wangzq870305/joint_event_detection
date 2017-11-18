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

from keras.preprocessing import sequence

from keras.layers import Input, Embedding, LSTM, Dense, merge, Merge
from keras.models import Model
from sklearn.metrics import average_precision_score
from keras.layers.core import Dense, Dropout, RepeatVector, Activation, Flatten

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

def get_lstm_input_output(part_name,vocab_size):
    main_input = Input(shape=(MAX_LEN,), dtype='int32', name=part_name+'_input')
        
    x = Embedding(output_dim=EMBED_SIZE, input_dim=vocab_size, input_length=MAX_LEN)(main_input)
        
    lstm_out = LSTM(HIDDEN_SIZE)(x)
    
    #lstm_out = x

    return main_input,lstm_out

def combined_train(X_train_list,y_train,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    
    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_lstm_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
        
    x = merge(out_list, mode='concat')
    
    #flatten = Reshape((180,)) (merged)

    #x= RepeatVector(HIDDEN_SIZE)(x)

    #x = LSTM(HIDDEN_SIZE)(x)
    
    main_loss = Dense(1, activation='sigmoid', name='main_output')(x)
    
    model = Model(input=input_list, output=main_loss)
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
      
    model.fit(X_train_list, y_train,nb_epoch=EPOCHS, batch_size=BATCH_SIZE)
   
    return model

def simple_joint_train(X_train_list,y_train_list,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    y_train_list=[np.array(y_train) for y_train in y_train_list]

    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_lstm_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
        
    x = merge(out_list, mode='concat')
    
    similar_loss = Dense(1, activation='sigmoid', name='similar_output')(x)
    summary_loss0=Dense(1, activation='sigmoid', name='summary_output_0')(out_list[0])
    summary_loss1=Dense(1, activation='sigmoid', name='summary_output_1')(out_list[1])
    event_loss0=Dense(1, activation='sigmoid', name='event_output_0')(out_list[0])
    event_loss1=Dense(1, activation='sigmoid', name='event_output_1')(out_list[1])
    
    model = Model(input=input_list, output=[similar_loss,summary_loss0,summary_loss1,event_loss0,event_loss1])
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
      
    model.fit(X_train_list, y_train_list, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)
   
    return model

def event_similar_train(X_train_list,y_train_list,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    y_train_list=[np.array(y_train_list[i]) for i in range(5) if i<1 or i>2]

    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_lstm_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
        
    x = merge(out_list, mode='concat')
    
    event_loss0=Dense(1, activation='sigmoid', name='event_output_0')(out_list[0])
    event_loss1=Dense(1, activation='sigmoid', name='event_output_1')(out_list[1])
    
    x=merge([x,event_loss0,event_loss1],mode='concat')
    similar_loss = Dense(1, activation='sigmoid', name='similar_output')(x)
    
    model = Model(input=input_list, output=[similar_loss,event_loss0,event_loss1])
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
      
    model.fit(X_train_list, y_train_list, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)
   
    return model

def similar_summary_train(X_train_list,y_train_list,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    y_train_list=[np.array(y_train) for y_train in y_train_list[:3]]

    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_lstm_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
        
    x = merge(out_list, mode='concat')
    
    similar_loss = Dense(1, activation='sigmoid', name='similar_output')(x)
    
    event_out0=merge([out_list[0],similar_loss],mode='concat')
    event_out1=merge([out_list[1],similar_loss],mode='concat')
    
    event_loss0=Dense(1, activation='sigmoid', name='event_output_0')(event_out0)
    event_loss1=Dense(1, activation='sigmoid', name='event_output_1')(event_out1)
    
    model = Model(input=input_list, output=[similar_loss,event_loss0,event_loss1])
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
      
    model.fit(X_train_list, y_train_list, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)
   
    return model

def joint_train(X_train_list,y_train_list,vocab_size):
    N=len(X_train_list)
        
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    y_train_list=[np.array(y_train) for y_train in y_train_list]

    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_lstm_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
    
    # Event loss
    event_loss0=Dense(1, activation='sigmoid', name='event_output_0')(out_list[0])
    event_loss1=Dense(1, activation='sigmoid', name='event_output_1')(out_list[1])
        
    # Similar loss
    x = merge(out_list, mode='concat')
    x=merge([x,event_loss0,event_loss1],mode='concat')
    similar_loss = Dense(1, activation='sigmoid', name='similar_output')(x)
    
    # Summary loss
    summ_out0=merge([out_list[0],event_loss0,similar_loss],mode='concat')
    summ_out1=merge([out_list[1],event_loss1,similar_loss],mode='concat')
    summary_loss0=Dense(1, activation='sigmoid', name='summary_output_0')(summ_out0)
    summary_loss1=Dense(1, activation='sigmoid', name='summary_output_1')(summ_out1)
    
    model = Model(input=input_list, output=[similar_loss,summary_loss0,summary_loss1,event_loss0,event_loss1])
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
      
    model.fit(X_train_list, y_train_list, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)
   
    return model

def event_summary_train(X_train_list,y_train_list,vocab_size):
    N=len(X_train_list)
        
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    y_train_list=[np.array(y_train) for y_train in y_train_list[1:]]

    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_lstm_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
    
    # Event loss
    event_loss0=Dense(1, activation='sigmoid', name='event_output_0')(out_list[0])
    event_loss1=Dense(1, activation='sigmoid', name='event_output_1')(out_list[1])
        
   
    # Summary loss
    summ_out0=merge([out_list[0],event_loss0,],mode='concat')
    summ_out1=merge([out_list[1],event_loss1],mode='concat')
    summary_loss0=Dense(1, activation='sigmoid', name='summary_output_0')(summ_out0)
    summary_loss1=Dense(1, activation='sigmoid', name='summary_output_1')(summ_out1)
    
    model = Model(input=input_list, output=[summary_loss0,summary_loss1,event_loss0,event_loss1])
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
      
    model.fit(X_train_list, y_train_list, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)
   
    return model
