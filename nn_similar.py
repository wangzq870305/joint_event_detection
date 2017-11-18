#! /usr/bin/env python
#coding=utf-8
from __future__ import division
from document import *
import nn
import nn_multi
from keras.preprocessing import sequence

NUM_TRAIN=1000

class NNSimilar:
    def __init__(self,trains):
        trains1=getTrains(trains,NUM_TRAIN)
        simTrains,self.V=getSimDocuments(trains1)
        
        X_sim_train,y_sim_train=formatK(simTrains,self.V)
        
        self.model=nn.lstm_train(X_sim_train,y_sim_train,len(self.V))
        
    def similar(self,source,target):
        if len(source)==0 or len(target)==0:
            return 0
        else:
            tests=[]
            words={}
            for w in source:
                words['#0_%s' %w]=1
            for w in target:
                words['#1_%s' %w]=1
            tests.append(CDocument2(words,False))
            X_test,y_test=formatK(tests,self.V)
            X_test = sequence.pad_sequences(X_test, maxlen=nn.MAX_LEN)            
            X_pred=self.model.predict(X_test)
            results=[result[0] for result in X_pred]
            
            return results[0]

class NNCombinedSimilar:
    def __init__(self,trains,V):
        trains1=getTrains(trains,NUM_TRAIN)
        self.V=V
        
        X_train_list,y_train_sim,y_train_summ0,y_train_summ1,y_train_event0,y_train_event1=getJointVectors(trains1,V)
        
        self.model=nn_multi.combined_train(X_train_list,y_train_sim,len(V))
        
    def similar(self,source,target):
        if len(source)==0 or len(target)==0:
            return 0
        else:
            tests_list=[[CDocument2(source,False)],[CDocument2(target,False)]]
            X_test_list=[]
            for tests in tests_list:
                X_test,y_test=formatK(tests,self.V)
                X_test = sequence.pad_sequences(X_test, maxlen=nn_multi.MAX_LEN)  
                X_test_list.append(X_test)
            
            X_pred=self.model.predict(X_test_list)
            results=[result[0] for result in X_pred] 
            return results[0]

class NNJointSimilar:
    def __init__(self,trains,V,train_function):
        trains1=getTrains(trains,NUM_TRAIN)
        self.V=V
        
        X_train_list,y_train_sim,y_train_summ0,y_train_summ1,y_train_event0,y_train_event1=getJointVectors(trains1,V)

        self.model=train_function(X_train_list,[y_train_sim,y_train_summ0,y_train_summ1,y_train_event0,y_train_event1],len(V))
        
    def similar(self,source,target):
        if len(source)==0 or len(target)==0:
            return 0
        else:
            tests_list=[[CDocument2(source,False)],[CDocument2(target,False)]]
            X_test_list=[]
            for tests in tests_list:
                X_test,y_test=formatK(tests,self.V)
                X_test = sequence.pad_sequences(X_test, maxlen=nn_multi.MAX_LEN)  
                X_test_list.append(X_test)
            
            X_pred=self.model.predict(X_test_list)
            results=[result[0] for result in X_pred[0]]

           # print X_pred
            return results[0]

class NNRank:
    def __init__(self,trains,V):
        train1=getTrains(trains,NUM_TRAIN)
        train1=getSummaryDocuments(train1)
        self.V=V
       
        X_train,y_train=formatK(train1,self.V)
       
        self.model=nn.lstm_train(X_train,y_train,len(self.V))
       
    def summarize(self,tests):
        X_test,y_test=formatK(tests,self.V)
        X_test = sequence.pad_sequences(X_test, maxlen=nn.MAX_LEN)            
        X_pred=self.model.predict(X_test)
        results=sorted([(result[0],i) for i,result in enumerate(X_pred)],reverse=True)
        
        return [tests[i] for score,i in results]

class NNJointRank:
    def __init__(self,model,V):
        self.model=model
        self.V=V
       
    def summarize(self,tests):
        X_test,y_test=formatK(tests,self.V)
        X_test = sequence.pad_sequences(X_test, maxlen=nn_multi.MAX_LEN)            
        X_pred=self.model.predict([X_test,X_test])
        results=sorted([(result[0],i) for i,result in enumerate(X_pred[1])],reverse=True)
        
        return [tests[i] for score,i in results]
