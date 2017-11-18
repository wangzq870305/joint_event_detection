#! /usr/bin/env python
#coding=utf-8

# get the random list from file
class CRandom:
    def __init__(self):
        self.a=[]
        for line in open(r'data/random-id.txt','rb'):
            self.a.append(int(line))
            
    def shuffle(self,b):
        n=len(b)
        return [b[i] for i in self.a if i<n]
        