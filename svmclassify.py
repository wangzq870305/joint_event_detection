#! /usr/bin/env python
#coding=utf-8
from __future__ import division
import subprocess
import math
from sklearn.metrics import average_precision_score

def getlexicon(documents):
    words=[]
    for document in documents:
        words+=document.words.keys()
    words=set(words)
    
    lexicon=dict([(word,i+1) for i,word in enumerate(words)])
    return lexicon
    
def createSvmText(documents,lexicon,path):
    text=''
    for document in documents:
        if document.polarity==True:
            line="+1 "
        else:
            line="-1 "
        pairs=[(lexicon[word],document.words[word]) for word in document.words.keys() if word in lexicon]
        pairs.sort()
        for pair in pairs:
            line+='%d:%d ' %(pair[0],pair[1])
        text+=line+'\n'
    if len(text)>0:
        output=open(path,'w')
        output.write(text)
     
def createResults(tests):
    input=open('result.output','rb')
    results=[]
    count=0
    for i,line in enumerate(input):
        score=float(line)
        if (tests[i].polarity==True and score>0) or (tests[i].polarity==False and score<0):
           count+=1 
        distance=float(line)
        x0=1/(1+math.exp(abs(distance)))
        x1=1/(1+math.exp(-1*abs(distance)))
        prob=x1/(x0+x1)
        if distance<0:prob*=-1
        results.append(prob)
#    acc=float(count)/len(tests)
#    print 'accuracy is %f(%d/%d)' % (acc,count,len(tests))
    
    p=n=tp=tn=fp=fn=0
    for i in range(len(tests)):
        score=results[i]
        if tests[i].polarity==True:
            p+=1
            if score>0:
                tp+=1
            else:
                fn+=1
        else:
            n+=1
            if score<0:
                tn+=1
            else:
                fp+=1

    acc=(tp+tn)/(p+n)
#    precisionP=tp/(tp+fp)
#    precisionN=tn/(tn+fn)
#    recallP=tp/(tp+fn)
#    recallN=tn/(tn+fp)
#    gmean=math.sqrt(recallP*recallN)
#    f_p=2*precisionP*recallP/(precisionP+recallP)
#    f_n=2*precisionN*recallN/(precisionN+recallN)
#    print '{gmean:%s recallP:%s recallN:%s} {precP:%s precN:%s fP:%s fN:%s} acc:%s' %(gmean,recallP,recallN,precisionP,precisionN,f_p,f_n,acc)
#    
#    # AUC
#    y=[]
#    py=[]
#    for i in range(len(tests)):
#        if tests[i].polarity==True:
#            y.append(1)
#        else:
#            y.append(0)
#        if results[i]>0:
#            py.append(results[i])
#        else:
#            py.append(abs(results[i]))
#
#    auc=average_precision_score(y, py)
#    print 'AUC %s' %auc
    #return (acc,f_p,f_n,auc),results
    
    print 'acc',acc
    
    return [],results
    
def svm_classify(trains,tests):
    lexicon=getlexicon(trains)
    createSvmText(trains,lexicon,'train.txt')
    createSvmText(tests,lexicon,'test.txt')
    subprocess.call("sh cmd.sh",shell=True)
    return createResults(tests)

def svm_learn(trains):
    lexicon=getlexicon(trains)
    createSvmText(trains,lexicon,'train.txt')
    subprocess.call("./svm_learn train.txt train.model > dump",shell=True)
    
    return lexicon

def svm_predict(tests,lexicon):
    createSvmText(tests,lexicon,'test.txt')
    subprocess.call("./svm_classify test.txt train.model result.output > dump ",shell=True)
    return createResults(tests)


def run_svm_classify(tests):
    subprocess.call("cmd.bat",shell=True)
    return createResults(tests)

