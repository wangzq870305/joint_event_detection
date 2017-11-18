#! /usr/bin/env python
#coding=utf-8
from __future__ import division
import math
from document import *
import random
from cluster_summarization import *

# Petrovic et al., (NAACL, 2010). Streaming first story detection with application to Twitter

t=0.5

SIM=Cosine()

def link(score):
    return 1-score<t

def nearest_neighbor_cluster_id(a,documents):
    if len(documents)>0:
        score,b=sorted([(SIM.similar(a.words,d.words),d)for d in documents],reverse=True)[0]
        if link(score):
            return b.clusterID
        else:
            return -1
    else:
        return -1
        
def LSH(documents):
    old_documents=[]
    clusters=[]
    for d in documents:
        cID=nearest_neighbor_cluster_id(d,old_documents)
        if cID==-1:
            cID=len(clusters)
            clusters.append(Cluster())

            d.clusterID=cID
            clusters[cID].add(d)
            old_documents.append(d)

        else:
            d.clusterID=cID
            clusters[cID].add(d)
            old_documents.append(d)
    
    for c in clusters:
            print len(c.elements)
        
    eval_purity(clusters)
    
    return clusters