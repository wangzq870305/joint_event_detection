#! /usr/bin/env python
#coding=utf-8
from __future__ import division
from document import *
from svmclassify import svm_classify

NUM_TRAIN=1000

def event_mention_classify(trains,unlabel):
    trains1=getTrains(trains,NUM_TRAIN)
    
    simTrains0=[]
    for d0,d1,simLabel,summaryLabel0,summaryLabel1,eventLabel0,eventLabel1 in trains1:
        simTrains0.append(CDocument2(d0.words,eventLabel0))
        simTrains0.append(CDocument2(d1.words,eventLabel1))
        
    pos=[]
    neg=[]
    for d in simTrains0:
        if d.polarity==True:
            pos.append(d)
        else:
            neg.append(d)
            
    print 'Event Mention',len(pos),len(neg)
    
    l=min([len(pos),len(neg)])
    
    eventTrains=pos[:l]+neg[:l]
    
    results=svm_classify(eventTrains,unlabel)[1]
    
    #return [unlabel[i] for i in range(len(unlabel)) if results[i]>0]
    
    newUnlabel=[]
    for i in range(len(unlabel)):
        prob=results[i]
        if prob<0:
            prob=1+prob
        
        if prob>0.3:
            newUnlabel.append(unlabel[i])
    return newUnlabel
    
        
