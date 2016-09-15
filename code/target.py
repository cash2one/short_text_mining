#-*- coding:utf-8 -*-
from __future__ import division
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class Target(object):


    def __init__(self, target, filter_words = None):
        
        #加载过滤字典
        if filter_words is not None:
            self.filter_words = self._file2set(fin = filter_words, index = 0)
        else:
            self.filter_words = set()

        self.target_list = target

        #target的个数
        self.label_num = set(target)

        #target的长度
        self.length = len(self.target_list)

    

    def _file2list(self, fin, index):
        with open(fin, "r") as f:
            key_list = [line.strip('\n').split('\t')[index] for line in f]
        return key_list

    def _file2set(self, fin, index):
        with open(fin, "r") as f:
            key_set = set([line.strip('\n').split('\t')[index] for line in f])
        return key_set
