#! /usr/bin/env python
#coding=utf-8
from document import *
from svmclassify import svm_classify
from cluster_summarization import clusterSummarization,Cosine,getSummaryForClusters,RandomSimilar,SVMSimilar,SVMRank
import nn_similar
from summary import PageRank,RandomRank
import nn_multi
from event import event_mention_classify
from lsh import LSH

LEN_OF_SUMMARY=10

def random_cluster(unlabel):
    ISim=RandomSimilar()
    return clusterSummarization(unlabel,20,ISim)

def cs_cluster(unlabel):
    ISim=Cosine()
    return clusterSummarization(unlabel,20,ISim)

def lsh_cluster(unlabel):
    return LSH(unlabel)

def lstm_cluster(trains,unlabel):
    ISim=nn_similar.NNSimilar(trains)
    return clusterSummarization(unlabel,20,ISim)

def random_summary(clusters,tests):
    ISumm=RandomRank()
    
    getSummaryForClusters(clusters,tests,ISumm,LEN_OF_SUMMARY)
    
def pagerank_summary(clusters,tests):
    ISumm=PageRank()
    
    getSummaryForClusters(clusters,tests,ISumm,LEN_OF_SUMMARY)
    
def lstm_summary(clusters,trains,tests,V):
    ISumm=nn_similar.NNRank(trains,V)
    
    getSummaryForClusters(clusters,tests,ISumm,LEN_OF_SUMMARY)
    
def JEDS(trains,tests,unlabel,V):
    ISim=nn_similar.NNJointSimilar(trains,V,nn_multi.joint_train)
    
    clusters=clusterSummarization(unlabel,20,ISim)
    
    #ISumm=nn_similar.NNJointRank(ISim.model,V)
    
    #getSummaryForClusters(clusters,tests,ISumm,LEN_OF_SUMMARY)
    