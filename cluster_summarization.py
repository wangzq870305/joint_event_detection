#! /usr/bin/env python
#coding=utf-8
from __future__ import division
import math
from document import *
import random
from svmclassify import svm_learn,svm_predict

NUM_TRAIN=3000

class Cluster:
    def __init__(self):
        self.elements=[]
        self.words={}
    
    def updateCentre(self):
        centre={}
        for document in self.elements:
            for word in document.words:
                if word not in centre:
                    centre[word]=0
                centre[word]+=1
        self.words=dict([(word,centre[word]/len(self.elements)) for word in centre])
        
    def add(self,d):
        self.elements.append(d)
        self.updateCentre()
    
    def getCentre(self):
        return self.words

class SimilarList:
    def __init__(self):
        self.a=[]
    
    def getAvgSD(self):
        n=len(self.a)
        if n>0:
            avg=sum(self.a)/n
            sd=math.sqrt(sum([(i-avg)**2 for i in self.a])/n)
            return avg,sd
        else:
            return 0,0
    
    def addRange(self,b):
        self.a+=[i for i in b if i>0]

class Cosine:
    def __init__(self):
        pass
    
    def similar(self,source,target):
        numerator=sum([source[word]*target[word] for word in source if word in target])
        sourceLen=math.sqrt(sum([value*value for value in source.values()]))
        targetLen=math.sqrt(sum([value*value for value in target.values()]))
        denominator=sourceLen*targetLen
        if denominator==0:
            return 0
        else:
            return numerator/denominator


class SVMSimilar:
    def __init__(self,trains):
        trains1=getTrains(trains,NUM_TRAIN)
        simTrains,self.V=getSimDocuments(trains1)
        
        self.lexicon=svm_learn(simTrains)
        
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
            
            result=svm_predict(tests,self.lexicon)[1][0]
            
            if result>0:
                return result
            else:
                return 1+result
            

class SVMRank:
    def __init__(self,trains):
        train1=getTrains(trains,NUM_TRAIN)
        train1=getSummaryDocuments(train1)
        
        self.lexicon=svm_learn(train1)
       
    def summarize(self,tests):
        results=svm_predict(tests,self.lexicon)[1]
        
        results=sorted([(result,i) for i,result in enumerate(results)],reverse=True)
        
        return [tests[i] for score,i in results]



class RandomSimilar:
    def __init__(self):
#        self.random=Random()
        pass

    def similar(self,source,target):
        return random.random()


def getMostSimilarCluster(clusters,d,sim):
    results=sorted([(sim.similar(clusters[i].getCentre(),d.words),i) for i in range(len(clusters))],reverse=True)
    similar,cIndex=results[0]
    currentSimilarList=[results[i][0] for i in range(len(results))]
    
    return similar,cIndex,currentSimilarList

def addToOldCluster(clusters,cIndex,d):
    c=clusters[cIndex]
    c.add(d)
    clusters.remove(c)
    clusters.insert(0,c)
    
def addToNewCluster(clusters,cIndex,d):
    c=Cluster()
    c.add(d)
    del clusters[-1]
    clusters.insert(0,c)
    
def eval_purity(clusters):
    
    # Purity
    newClusters=[c for c in clusters if len(c.elements)>10]

    m=sum([len(c.elements) for c in newClusters])
    p=0
    for c in newClusters:
        n=len(c.elements)
        tags={}
        for d in c.elements:
            if d.id not in tags:
                tags[d.id]=0
            tags[d.id]+=1
        mostFreqCount=sorted([count for count in tags.values()],reverse=True)[0]
        score=mostFreqCount/n
        p+=(n/m)*score
    
    print 'purity:',p
    
    # C_min
    C_miss=0.5
    C_FA=0.5
    P_miss=0
    P_FA=0
    for i,c in enumerate(newClusters):
        n=len(c.elements)
        fa=0
        miss=0
        
        tags={}
        for d in c.elements:
            if d.id not in tags:
                tags[d.id]=0
            tags[d.id]+=1
        tcount,tid=sorted([(count,id) for id,count in tags.items()],reverse=True)[0]
        
        for d in c.elements:
            if d.id!=tid:
                fa+=1
        
        for j in range(len(newClusters)):
            if j!=i:
                for d in newClusters[j].elements:
                    if d.id==tid:
                        miss+=1
        fa=fa/len(c.elements)
        miss=miss/(miss+tcount)
        
        P_FA+=fa
        P_miss+=miss
        
    #print 'P_FA',P_FA/len(newClusters)
    #print 'P_Miss',P_miss/len(newClusters)
    print 'C_min',C_FA*P_FA/len(newClusters)+C_miss*P_miss/len(newClusters)
        

def clusterSummarization(documents,k,sim):
    clusters=[Cluster() for i in range(k)]
    similarList=SimilarList()
    avg=0
    sd=0
    
    for i,d in enumerate(documents):
        similar,cIndex,currentSimilarList=getMostSimilarCluster(clusters,d,sim)
        similarList.addRange(currentSimilarList)
        #print similar,avg-3*sd
        if similar<avg:
            addToNewCluster(clusters,cIndex,d)
        else:
            addToOldCluster(clusters,cIndex,d)
        avg,sd=similarList.getAvgSD()
        #print avg,sd
    
    for c in clusters:
        print len(c.elements)
    
    eval_purity(clusters)
    
    return clusters
    
def getSummaryForClusters(clusters,tests,summ,LEN_OF_SUMMARY):
    newClusters=[c for c in clusters if len(c.elements)>10]
    
    eventDict={}
    for i,event in enumerate(tests):
        pos,neg,id,goldData=event
        eventDict[id]=goldData
    
    resultDict={}
    for c in newClusters:
        n=len(c.elements)
        tags={}
        allDocuments=[]
        for d in c.elements:
            if d.id not in tags:
                tags[d.id]=[]
            tags[d.id].append(d)
            allDocuments.append(d)
        id,documents=sorted([(len(documents),id,documents) for id,documents in tags.items()],reverse=True)[0][1:]
        print id
        #resultDict[id]=documents
        resultDict[id]=allDocuments # since we don't know the id of event, we need use all the documents as input for summary
    
    clearDir(r'../rouge-summary/summaries-gold/')
    clearDir(r'../rouge-summary/summaries-system/pr/')
    
    for id in resultDict:
        documents=resultDict[id]
        testResult=summ.summarize(documents)
        
        goldData=eventDict[id]
        writeGoldData(id,goldData)
        writeTestData(id,testResult[:LEN_OF_SUMMARY])
        

