#! /usr/bin/env python
#coding=utf-8
import os
import shutil
import numpy as np
from crandom import CRandom

MONTHS=set(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])

class Document:
    def __init__(self,text,words):
        self.text=text
        self.words=words

class CDocument:
    def __init__(self,text,words,polarity,id):
        self.text=text
        self.words=words
        
        self.polarity=polarity
        self.id=id
        
        # for Keras
        if polarity==True:
            self.label=1
        else:
            self.label=0

class CDocument2:
    def __init__(self,words,polarity):
        self.words=words
        
        self.polarity=polarity
        
        # for Keras
        if polarity==True:
            self.label=1
        else:
            self.label=0

def isEvent(text,id):
    p=id.split('_')
    keywords=set([w.lower() for w in p if w.isdigit()==False and w not in MONTHS])
    if len([w for w in text.split() if w.lower() in keywords])>0:
        return 1
    else:
        return 0
        

def readGoldData():
    dir_path=r'data/golds'
    
    documents=[]
    for fpath in os.listdir(dir_path):
        id=fpath
        lines=[]
        for line in open(dir_path+'//'+fpath,'rb'):
            lines.append(line.strip())
        documents.append((id,' '.join(lines))) # (id, news)
    return dict(documents)
    
def readTestData():
    dir_path=r'data/tests'
    
    documents=[]
    for fpath in os.listdir(dir_path):
        id=fpath[:-4]
        lines=[]
        for line in open(dir_path+'//'+fpath,'rb'):
            lines.append(line.strip())
        documents.append((id,lines)) # (id, tweets)
    return documents
    
    
def getData():
    gold=readGoldData()
    test=readTestData()
    
    print 'length of gold and test data:',len(gold),len(test)
    data=[(id,gold[id],tweets) for id,tweets in test if id in gold]

    return data

def unigram(text):
    words={}
    for w in text.split():
        words[w.lower()]=1
    return words

def writeGoldData(id,goldData):
    dir=r'../rouge-summary/summaries-gold/A%sA' %id
    
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    
    for i in range(3):
        file_path=dir+r'/A%sA.%d.gold' %(id,i+1)
        output=open(file_path,'w')
        output.write(goldData)

def writeTestData(id,results):
    file_path=r'../rouge-summary/summaries-system/pr/A%sA.pr.system' %id
    output=open(file_path,'w')
    for r in results:
        output.write('%s\n' %r.text)

def clearDir(dir):
    if os.path.exists(dir):
        print 'clear dir: %s' %dir
        shutil.rmtree(dir)
    os.mkdir(dir)

def formatK(data,V):
    X=[]
    Y=[]
    
    for i,d in enumerate(data):
        x=[]
        for w in d.words:
            if w in V:
                if d.words[w]>=1:
                    x.append(V[w])
        y=d.label
        X.append(x)
        Y.append(y)
    return X,Y

def formatText(text,V):
    X=[]
    for w in text.split():
        if w in V:
            X.append(V[w])
    return X

def getVocabrary_df(data,k=100):
    # DF
    df={}
    for id,goldData,tweets in data:
        for w in goldData.split():
            if w not in df:
                df[w]=0
            df[w]+=1
        for t in tweets:
            for w in t.split():
                if w not in df:
                    df[w]=0
                df[w]+=1

    df=sorted([(df[w],w) for w in df])
    df.reverse()
    df=[w for count,w in df]

    # feature selection
    V={}
    for i,w in enumerate(df):
        if i<(k): # make sure the feature space contain some English words
            V[w]=len(V)

    print 'length of V:',len(V)

    return V
        
def getCEvents(data):
    CEvents=[]
    for id,goldData,tweets in data:        
        # pos - gold data
        sentences=goldData.split('.')
        sentences=set([sentence.strip() for sentence in sentences if len(sentence.split())>2])
        gold_sentences=[sentence for sentence in sentences]
        
        # neg - tweets
        sentences=set([sentence.strip() for sentence in tweets if len(sentence.split())>2])
        test_sentences=[sentence for sentence in sentences]
        print len(gold_sentences),len(test_sentences)
        
        if len(gold_sentences)>0 and len(test_sentences)>0:
            minN=min([len(gold_sentences),len(test_sentences)])
            pos=[CDocument(sentence,unigram(sentence),True,id) for sentence in gold_sentences]
            neg=[CDocument(sentence,unigram(sentence),False,id) for sentence in test_sentences]
            #print pos,neg
            CEvents.append((pos,neg,id,goldData))
    return CEvents

def getCDocuments(events,V,isTrains):
    documents=[]
    for pos,neg,id,goldText in events:
        if isTrains:
            minN=min([len(pos),len(neg)])
            documents+=pos[:minN]+neg[:minN]
        else:
            documents+=neg
    
    for d in documents:
        for w in d.words.keys():
            if w not in V:
                del d.words[w]        
    
    return documents

def writeResults(tests,results,LEN_OF_SUMMARY):
    clearDir(r'../rouge-summary/summaries-gold/')
    clearDir(r'../rouge-summary/summaries-system/pr/')
    
    k=0
    for i,event in enumerate(tests):
        pos,neg,id,goldData=event
        eResults=[]
        for j in range(len(neg)):
            prob=results[k]
            if prob<0:
                prob=1+prob
            eResults.append((prob,j))
            k+=1
        eResults.sort()
        eResults.reverse()
   
        testResult=[neg[j] for prob,j in eResults]
        writeGoldData(id,goldData)
        writeTestData(id,testResult[:LEN_OF_SUMMARY])

def getEventCount(documents):
    pCount=0
    nCount=0
    
    pECount=0
    nECount=0
    
    for d in documents:
        if d[3]==True:
            pCount+=1
        else:
            nCount+=1
        if isEvent(d[0].text,d[0].id)>0:
            pECount+=1
        else:
            nECount+=1
        
    print 'summary:',pCount,nCount
    print 'event:',pECount,nECount
    
def getX(d,V):
    x=[]
    for w in d.words:
        if w in V:
            if d.words[w]>=1:
                x.append(V[w])
    return x

def getTrains(trains,n):
    documents=[]
    for pos,neg,id,goldText in trains:  
        documents+=pos
        documents+=neg 
    
    N=len(documents)

    simPos=[]
    simNeg=[]
    for i in range(N):
        for j in range(i+1,N):
            summaryLabel0=0
            if documents[i].polarity:
                summaryLabel0=1
            summaryLabel1=0
            if documents[j].polarity:
                summaryLabel1=1
            
            if documents[i].id==documents[j].id: # from same event, similar posts
                simPos.append((documents[i],documents[j],True,summaryLabel0,summaryLabel1,isEvent(documents[i].text,documents[i].id),isEvent(documents[j].text,documents[j].id)))
            else:
                simNeg.append((documents[i],documents[j],False,summaryLabel0,summaryLabel1,isEvent(documents[i].text,documents[i].id),isEvent(documents[j].text,documents[j].id)))
    
    # random
    cRandom=CRandom()
    simPos=cRandom.shuffle(simPos)
    simNeg=cRandom.shuffle(simNeg)
    
    print 'similar:',len(simPos),len(simNeg)
    getEventCount(simPos[:n]+simNeg[:n])
    
    return simPos[:n]+simNeg[:n]

def getSimDocuments(trains):
    V={}
    simTrains0=[]
    for d0,d1,simLabel,summaryLabel0,summaryLabel1,eventLabel0,eventLabel1 in trains:
        words={}
        for w in d0.words:
            words['#0_%s' %w]=1
        for w in d1.words:
            words['#1_%s' %w]=1
        for w in words:
            if w not in V:
                V[w]=len(V)
        simTrains0.append(CDocument2(words,simLabel))
    
    return simTrains0,V

def getSummaryDocuments(trains):
    simTrains0=[]
    for d0,d1,simLabel,summaryLabel0,summaryLabel1,eventLabel0,eventLabel1 in trains:
        simTrains0.append(CDocument2(d0.words,summaryLabel0))
        simTrains0.append(CDocument2(d1.words,summaryLabel1))
        
    return simTrains0

def getEventDocuments(trains):
    simTrains0=[]
    for d0,d1,simLabel,summaryLabel0,summaryLabel1,eventLabel0,eventLabel1 in trains:
        simTrains0.append(CDocument2(d0.words,eventLabel0))
        simTrains0.append(CDocument2(d1.words,eventLabel1))
        
    return simTrains0


def getJointVectors(trains,V):
    X0=[]
    X1=[]
    
    YSim=[]
    YSumm0=[]
    YSumm1=[]
    YEvent0=[]
    YEvent1=[]
    for d0,d1,simLabel,summaryLabel0,summaryLabel1,eventLabel0,eventLabel1 in trains:
        x0=getX(d0,V)
        x1=getX(d1,V)
        X0.append(x0)
        X1.append(x1)
        YSim.append(simLabel)
        YSumm0.append(summaryLabel0)
        YSumm1.append(summaryLabel1)
        YEvent0.append(eventLabel0)
        YEvent1.append(eventLabel1)
                
    return [X0,X1],YSim,YSumm0,YSumm1,YEvent0,YEvent1
