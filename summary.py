#! /usr/bin/env python
#coding=utf-8
from __future__ import division
import numpy as ny
import math
import random

#import sentiment

# consine similarity process
# calculate cosine similarity of two vector
# cosine(s,t)=(s*t)/(|s|*|t|)
#
# @para
# source-source vector
# target-target vector
def cosine(source,target):
    numerator=sum([source[feature]*target[feature] for feature in source if feature in target])
    sourceLen=math.sqrt(sum([value*value for value in source.values()]))
    targetLen=math.sqrt(sum([value*value for value in target.values()]))
    denominator=sourceLen*targetLen
    if denominator==0:
        return 0
    else:
        return numerator/denominator

# pagerank process
#
# ref. 
# [1] Wan X. and Yang J. Multi-Document Summarization Using Cluster-Based Link Analysis. In Proceeding of SIGIR-08.
# [2] Page L., S. Brin, R. Motwani and T. Winograd. The pagerank citation ranking: Bringing order to the web. Technical report. 1998.
#
# pr=0.15+0.85*sum(pr[j]*link[i,j])
# 
# @para
# T-transfer matrix
# m-length of X
# iter_num-iteration times
# 
# @return
# the sorted indexs of vectors after pagerank ranking process
def pagerank(T,m,iter_num=10):    
    # init Y
    Y=[1 for i in range(m)]
    
    # normalization T
    NT=ny.zeros((m,m))
    for i in range(m):
	sumJ=sum([T[i,j] for j in range(m)])
	for j in range(m):
	    NT[i,j]=T[i,j]/sumJ
    
    # PageRank process
    for t in range(iter_num):
	# temp of Y
        tY=[0 for i in range(m)]
        for i in range(m):
            tY[i]=0.15+0.85*sum([NT[j,i]*Y[j] for j in range(m) if i!=j])  
        Y=tY
        
        print t
    
    results=[(y,i) for i,y in enumerate(Y)]
    
    # sort results
    results.sort()
    results.reverse()
    return [result[1] for result in results]

# redundance control
# remove similar sample
# 
# @para
# T-transfer matrix
# m-length of X
# iter_num-iteration times
# threshold-similar threshold
# 
# @return
# the sorted indexs of vectors after redundance control process
def redundance_control(T,tests,m,pr_results,threshold=0.75):
    # index-sort_index map of pagerank results
    pr_results_dict=dict([(index,sort_index) for sort_index,index in enumerate(pr_results)])
    
    # redundance control process
    redundance=[]
    for i in range(m):
	for j in range(i+1,m):
	    # if the similarity of two vector is too large, remove one vector with low rank score of pagerank process
	    if T[i,j]>threshold:
		if pr_results_dict[i]>pr_results_dict[j]:
		    redundance.append(j)
		else:
		    redundance.append(i)   
		    
    return [i for i in pr_results if i not in set(redundance)]

# find clause similar sample
def clauseSimilar(t1,t2):
    c1=t1.split(u'，')
    c2=t2.split(u'，')
    
    for cc1 in c1:
        for cc2 in c2:
            cc1=cc1.strip()
            cc2=cc2.strip()
            if cc1==cc2:
                return True
    return False
    
# remove repeat samples
def removeRepeat(tests):
    indexs=[]
    d={}
    for i in range(len(tests)):
        if (tests[i].text not in d) and ('  '.join(tests[i].text.split()[:-1]) not in d):
            d[tests[i].text]=1
            indexs.append(i)
    print len(indexs)
    return [tests[i] for i in indexs]

# text summarize process
#
# @para
# tests-the vector of tests
# 
# @return
# the sorted indexs of tests vectors after summaize process
def pr_summarize(tests):
    X=[v.words for v in tests]
    m=len(X)
    
    print '===transfer matrix==='
    #document-to-document transfer matrix
    T=ny.zeros((m,m))
    for i in range(m):
        for j in range(m):
            T[i,j]=cosine(X[i],X[j])   
        print i
	    
    pr_results=pagerank(T,m)
    rc_results=pr_results
    rc_results=redundance_control(T,tests,m,pr_results,threshold=0.5)
    return [tests[i] for i in rc_results]


class PageRank:
    def __init__(self):
        pass
    
    def summarize(self,tests):
        return pr_summarize(tests)

class RandomRank:
    def __init__(self):
        pass

    def summarize(self,tests):
        random.shuffle(tests)
        return tests
