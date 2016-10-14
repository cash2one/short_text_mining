#-*- coding:utf-8 -*-
import data
import feature
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from tool import token_split
import pickle

class Model(object):

    
    def __init__(self):
        pass

    def set_train_params(self, input, feat_num, target, stop_words = None):
        #加载原始数据
        data_inst = data.Data(stop_words = stop_words)
        data_inst.load_data(fin = input, sep = "\t", key_dict={"title":0,"target":2,"feature":1}, number = 3)
        data_inst.transformBinary(target = target)
        self.data_inst = data_inst

        #初始化变量
        self.target = target
        self.feat_num = feat_num



    def get_feature(self, sep, algorithm):
        #特征选择
        feature_inst = feature.Feature()
        feature_inst.data2vector(data = self.data_inst, sep = sep)
        feature_inst.select_feat(feat_num = self.feat_num, algorithm = algorithm)
        self.feat_names = feature_inst.select_feat_name
        self.feat_inst = feature_inst


    def predict_prob_label(self, data_inst):
        print("{0}".format(self.clf.classes_))
        X_data = self.vector.fit_transform(data_inst.feature)
        prob_label = self.clf.predict_proba(X_data)

        #构建DataFrame,返回结果
        return pd.DataFrame({"title":pd.Series(data_inst.title),
                             "feature":pd.Series(data_inst.feature),
                             ("category"+self.clf.classes_[0]):pd.Series(prob_label[:,0]),
                             ("category"+self.clf.classes_[1]):pd.Series(prob_label[:,1]),
                             "target":pd.Series(data_inst.target)
                             })

        
        

        
    def transform_train_data(self, ratio):
    
        #调整正负样例比例
        if ratio == "all":
            self.feature_list = self.data_inst.feature
            self.target_list = self.data_inst.target
        else:
        
            sample_df = pd.DataFrame({})
            #获取正样例数组
            data_df = pd.DataFrame({"target":pd.Series(self.data_inst.target),
                                    "feature":pd.Series(self.data_inst.feature)})

            positive_df = data_df[data_df.target == "1"]
            negative_df = data_df[data_df.target == "0"]

            positive_df_num = len(positive_df.index)
            negative_df_num = len(negative_df.index)

            print("Before sample, positive number: {0}, negative number: {1}".format(positive_df_num, negative_df_num))
            
            #添加正样本样例
            sample_df = sample_df.append(positive_df)
            if positive_df_num >= negative_df_num:
                sample_df = sample_df.append(negative_df)
            else:
                sample_df = sample_df.append(negative_df.sample(n = int(positive_df_num * ratio)))
            print("After sample, positive number: {0}, negative number: {1}".format(len(sample_df[sample_df.target == "0"].index), len(sample_df[sample_df.target == "1"].index)))
            self.feature_list = sample_df.feature.tolist()
            self.target_list = np.array(sample_df.target)

    def fit(self, ratio, sep, select_alg, clf_alg):
        
        #特征选择
        self.get_feature(sep = sep, algorithm = select_alg)
        
        #调整正负样例比例，获取训练样本
        self.transform_train_data(ratio = ratio)
    
        #训练模型
        vocabulary = self.feat_names
        
        self.vect = CountVectorizer(vocabulary = vocabulary, tokenizer = token_split)
        X_train = self.vect.fit_transform(self.feature_list)
        y_train = self.target_list
        self.clf = clf_alg(C = 100, penalty='l1', tol=0.01)
        self.clf.fit(X_train, y_train)
        print("sccore with L1 penalty: %.4f" %(self.clf.score(X_train, y_train)))

    def persistence_vector(self, vector_fout):
        joblib.dump(self.vect, vector_fout)

    def persistence_clf(self, clf_fout):
        joblib.dump(self.clf, clf_fout)


    def selectFeat2file(self, feat_fout):
        self.feat_inst.feat2file(file = feat_fout)


    def load_vector(self, vector_fout):
        vector = joblib.load(vector_fout)
        vocabulary = vector.vocabulary_
        self.vector =  CountVectorizer(vocabulary = vocabulary, tokenizer = token_split)
        print("vector: {0}".format(self.vector))

    def load_clf(self, clf_fout):
        self.clf = joblib.load(clf_fout)
        print("clf: {0}".format(self.clf))
       
        






if __name__ == "__main__":

    import sys
    if len(sys.argv) != 8:
        sys.exit("Usage: input feat_num feat_fout vectorizer model target stop_words")
        
    input = sys.argv[1]
    feat_num = int(sys.argv[2])
    feat_fout = sys.argv[3]
    vector_fout = sys.argv[4]
    clf_fout = sys.argv[5]
    target = sys.argv[6]
    stop_words = sys.argv[7]

    model_inst = Model()
    model_inst.set_train_params(input = input, feat_num = feat_num, target = target, stop_words = stop_words)
    #训练模型
    model_inst.fit(ratio = 0.6, sep = ",", select_alg = chi2, clf_alg = LogisticRegression)
    #固化向量化模型
    model_inst.persistence_vector(vector_fout = vector_fout)
    
    #固化分类模型
    model_inst.persistence_clf(clf_fout = clf_fout)
    #保存特征向量
    model_inst.selectFeat2file(feat_fout = feat_fout)


    #加载向量化模型
    model_inst.load_vector(vector_fout = vector_fout)

    #加载向量化特征向量
    model_inst.load_clf(clf_fout = clf_fout)


