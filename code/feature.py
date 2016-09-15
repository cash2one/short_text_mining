#-*- coding:utf-8 -*-
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class Feature(object):


    def __init__(self, feats, stop_words = None):
        
        if stop_words is not None:
            self.stop_words = self._file2set(fin = stop_words, index = 0)
        else:
            self.stop_words = set()

        #加载特征数据
        self.feat_list = self.feats2list(feats = feats)

        #加载特征字符串数据
        self.str_list = feats

        
            

    def _file2list(self, fin, index):
        with open(fin, "r") as f:
            key_list = [line.strip('\n').split('\t')[index] for line in f]
        return key_list

    def _file2set(self, fin, index):
        with open(fin, "r") as f:
            key_set = set([line.strip('\n').split('\t')[index] for line in f])
        return key_set


    def feats2list(self, feats):
        feat_list = []
        for feat in feats:
            field_list = [field for field in feat.strip().split(",") if field not in self.stop_words]
            feat_list.append(field_list)
            
        return feat_list
                


    def data2vector(self, data, sep):
        count_vector = CountVectorizer(stop_words = self.stop_words,tokenizer=lambda x: x.split(sep))
        self.vect_data = count_vector.fit_transform(data.feature)
        self.target = data.target
        self.feat_names = count_vector.get_feature_names()

    def select_feat(self, feat_num, algorithm):
        
        feat_kbest = SelectKBest(algorithm, k=feat_num)
        feat_kbest.fit_transform(self.vect_data, self.target)
        self.select_feat_name = [self.feat_names[i] for i in feat_kbest.get_support(indices = True)]
        self.select_feat_score = [feat_kbest.scores_[i] for i in feat_kbest.get_support(indices = True)]
        
    def featScore2file(self, file):
        """计算每个feature和对应的score得分


        """
        with open(file, 'w') as f:
            for i in range(len(self.select_feat_name)):
                f.write("{0}, {1}\n".format(self.select_feat_name[i].encode("utf-8"), str(self.select_feat_score[i])))

    def feat2file(self, file):
        """将每条记录的feature保存到指定文件中

        """
        with open(file, 'w') as f:
            for feats in self.feat_list:
                f.write("{}\n".format(",".join(feats)))



    
