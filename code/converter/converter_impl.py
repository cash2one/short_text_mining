# !/usr/bin/env python
#-*- coding:utf-8 -*-

__all__ = ["Data", "Text2countConverter", "convert_text"]


import sys
import numpy as np
import unicodedata
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer

class Text2vectConverter(object):
    """
    """

    def __init__(self, option = ""):
        self._option = option


    def _parese_option(self, option):
        text_prep_opt, feat_gen_opt, label_prep_opt = '', '', ''
        option = option.strip().split()
        
        i = 0
        while i < len(option):
            if i + 1 >= len(option):
                raise ValueError("{} can not be the last option.".format(option[i]))
            
            if type(option[i+1]) is not int and not option[i+1].isdigit():
                raise ValueError("Invalid option {} {}".format(option[i], option[i+1]))
            
            #stopword代表是否采用停用词;convert代表采用的转化类型
            if option[i] in ("-stopword", "-convert" ):
                text_prep_opt = " ".join([text_prep_opt, option[i], option[i+1]])
            elif option[i] in ("-feature"):
                feat_prep_opt = " ".join([feat_gen_opt, option[i], option[i+1]])
            elif option[i] in ("-binary"):
                label_prep_opt = " ".join([label_prep_opt, option[i], option[i+1]])
            else:
                raise ValueError("The option {} not in text_prep_opt or feat_prep_opt or label_prep_opt.".format(option[i]))
            i += 2
        return text_prep_opt, feat_gen_opt, label_prep_opt
       
 
  


class Data(object):

    def __init__(self):
        pass
        

    def _file2dict(self, fin):
        with open(fin, 'r') as f:
            list = [line.strip('\n') for line in f]
        return dict(zip(list, range(1,len(list)+1)))

    def _uniCode(self, text):
        text = unicodedata.normalize('NFD', unicode(text, 'utf-8'))
        return text

    def train_test_split(self, split_ratio, fout_train, fout_test):
        """Get the train data and test data 

        Split the source data into the train data and test data based
        the specified split_ratio(test_number/total_number)

        Args:
            split_ratio: float type, split ratio(test_number/total_number)
            fout_train: string type, the output file name for train data
            fout_test: string type, the output file name for test data
        """

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(self.features, 
                                                                             self.targets, 
                                                                             test_size = split_ratio,
                                                                             random_state = 0)

        #save the train data
        self._data2file(datasets = X_train, labels = y_train, fout = fout_train)

        #save the test data
        self._data2file(datasets = X_test, labels = y_test, fout = fout_test)

    
    def _data2file(self, datasets, labels, fout):
        with open(fout, 'w') as f:
            for index, label in enumerate(labels):
                f.write("{}\t{}\n".format(labels[index].encode("utf-8"), datasets[index].encode("utf-8")))



    def load_data(self, fin, sep, number, **kwargs):
        """load data

        Split the each line in fin to the specified fields, and 
        save them in to the list
     
        Args:
            fin: string type, the input file name
            sep: string type, the separator between label and feature
            number: int type, the number of splitted fields
        """

        if "target" in kwargs:
            target_list = []
        
        if "feature" in kwargs:
            feature_list = []

        if "title" in kwargs:
            title_list = []


        #record the number of samples
        sample_num = 0

            
        with open(fin, "r") as f:
            for line in f:
                fields = line.strip().split(sep)
                if len(fields) != number:
                    print("[Data object]-[load_train_data]-[Warning]: split number is wrong: {}".format(line))
                    continue
                
                sample_num += 1
                for key in kwargs:
                    if key == "target":
                        target_list.append(self._uniCode(fields[kwargs[key]]))
                        continue

                    if key == "feature":
                        feature_list.append(self._uniCode(fields[kwargs[key]]))
                        continue

                    if key == "title":
                        title_list.append(self._uniCode(fields[kwargs[key]]))
                        continue


        if "target" in kwargs and len(target_list) != sample_num:
            sys.exit("[Data object] [load_train_data]-[Error]: the number of target is not equal to sample_num")
        else:
            self.targets = target_list
            
        if "feature" in kwargs and len(feature_list) != sample_num:
            sys.exit("[Data object] [load_train_data]-[Error]: the number of feature is not equal to sample_num")
        else:
            self.features = feature_list
            
        if "title" in kwargs and len(title_list) != 0 and len(title_list) != sample_num:
            sys.exit("[Data object] [load_train_data]-[Error]: the number of title is not equal to sample_num")
        else:
            self.titles = title_list

        if sample_num != 0:
            self.total = sample_num

    def transformBinary(self, target):
        self.target[self.targets != target] = "0"
        self.target[self.targets == target] = "1"





        
    
                    
                        
                
