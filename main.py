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
from test_models import *

LEN_OF_TRAIN=10


data=getData()
CEvents=getCEvents(data) # (pos,neg,id,gold-text)
print len(CEvents)

V=getVocabrary_df(data,k=1000)

trains=CEvents[:LEN_OF_TRAIN]
tests=CEvents[LEN_OF_TRAIN:]

unlabel=[]
for pos,neg,id,goldText in tests:
    unlabel+=neg

# Filter
#print len(unlabel)
#unlabel=event_mention_classify(trains,unlabel)
#print len(unlabel)

# Cluster
#random_cluster(unlabel)

JEDS(trains,tests,unlabel,V)
