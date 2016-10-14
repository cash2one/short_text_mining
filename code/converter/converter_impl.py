# !/usr/bin/env python
#-*- coding:utf-8 -*-

__all__ = ["Data", "Text2countConverter", "convert_text"]

import sys
import numpy as np
import unicodedata
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureGenerator(object):
    """ Process Feature Procedure

    """

    def __init__(self, option = "-featSelection 0 -convert CountVectorizer"):
        """

        """

        self._option = option
        args = self._parse_option(option = option)

        #initialize vectorization procedure
        self.convert = args[0]
        #initialize feature selection procedure
        self.feat_select = args[1]

    @staticmethod
    def count_convert():
        """Get CountVectorizer instance

        """

        count_vector = CountVectorizer(tokenizer=lambda x: x.split(","),
                                       analyzer = 'word',
                                       binary = True)
        return count_vector

    @staticmethod
    def tfidf_convert():
        """Get TfidfVectorizer instance

        """

        tf_idf_vector = TfidfVectorizer(tokenizer=lambda x: x.split(","),
                                       analyzer = 'word',
                                       smooth_idf = False)
        return tf_idf_vector

    def _parse_option(self, option):
        """ Parese options

        """

        i = 0
        while i < len(option):
            if option[i] =="-convert":
                if option[i+1] == u'CountVectorizer':
                    text_convert = self.count_convert()
                elif option[i+1] == u'TfidfVectorizer':
                    text_convert = self.tfidf_convert()
                else:
                    raise ValueError("text_convert does not support {} convert.".format(option[i+1]))
            elif option[i] == "-featSelection":
                if int(option[i+1]) == '0':
                    feat_selection = None
                else:
                    raise ValueError("feat_selection does not support {} type.".format(option[i+1]))
            else:
                raise ValueError("parse_option does not support {} parameters.".format(option[i]))
            i += 2
        return text_convert, feat_selection
             
    def transform(self, feats):
        """Transform list to the specifial type vector
 
        """
        if feats:
            feat_vects = self.convert.fit_transform(feats)
        else:
            raise ValueError("feats is empty. ")
        return feat_vects
    
            
                

class Text2vectConverter(object):
    """
    """

    def __init__(self, option = ""):
        self._option = option
        #get configuration parameters
        text_prep_opt, feat_gen_opt, label_prep_opt = self._parse_option(option)
        
        #The TextPreprocesss instance for preprocessing data for classifier
        self.text_prep = TextPreprocessor(text_prep_opt)
        #The FeatureGenerator instance for preprocessing feature for classifier
        self.feat_gen = FeatureGenerator(feat_gen_opt)
        
    def transform(self, text_src, has_label = False, extra_feats = []):
        """
        """

        feats_list, labels_list = self.text_prep.preprocess(text_src = text_src, has_label = False)
        #process feature source
        feat_vect = self.feat_gen.transform(feats = feats_list)

        return feats_vect, labels_list
        
    def _parese_option(self, option):
        text_prep_opt, feat_gen_opt, label_prep_opt, tokenizer_opt = '', '', '', ''
        options = option.strip().split()
       
        
        i = 0
        while i < len(option):
            if i + 1 >= len(option):
                raise ValueError("{} can not be the last option.".format(option[i]))
            
            if type(option[i+1]) is not int and not option[i+1].isdigit():
                raise ValueError("Invalid option {} {}".format(option[i], option[i+1]))
            
            #stopword代表是否采用停用词;convert代表采用的转化类型
            if option[i] in ("-stopword"):
                text_prep_opt = " ".join([text_prep_opt, option[i], option[i+1]])
            elif option[i] in ("-feature", "-convert"):
                feat_prep_opt = " ".join([feat_gen_opt, option[i], option[i+1]])
            elif option[i] in ("-binary"):
                label_prep_opt = " ".join([label_prep_opt, option[i], option[i+1]])
            else:
                raise ValueError("The option {} not in text_prep_opt or feat_prep_opt or label_prep_opt.".format(option[i]))
            i += 2
        return text_prep_opt, feat_gen_opt, label_prep_opt
       
class TextPreprocessor(object):

    def __init__(self, option):
        self.option = option
        opts = self._parse_option(option)
        #set stop words
        self.stopword_remover = opts[0]
       
    def _parse_option(self, option = "-stopword 0 -convert CountVectorizer"):
        i = 0
        while i < len(option):
            if option[i][0] != '-': break
            if option[i] == '-stopword':
                if int(option[i+1]) != 0:
                    stopword_set = self.default_stopword()
                else:
                    stopword_set = set()
            else:
                raise ValueError("parse_option for TextPreprocessor can not proprecsss {} parameters.".format(option[i]))
            i += 2
        return stopword_set

    def preprocess(self, text_src, has_label=True):
        """Preprocess text_src data into vector

        """
        
        if not isinstance(text_src, str):
            raise ValueError("text_src type is {}, but it is not string type".format(type(text_src)))

        feats = []
        labels = []
        with open(text_src, 'r') as f:
            for line in f:
                fields = line.strip("\r\n").split("\t")
                words = ",".join([word for word in fields[0].strip().split(",") if word not in self.stopword_set])
                if words:
                    feats.append(words)
                    if has_label:
                        labels.append(fields[1])

        return feats, labels

    @staticmethod
    def default_stopword():
        """Get stop words set

        """
        stopwords = set()
        src = ""
        if not src:
            src = "{0}/stop-words/stop_words.dat".format(os.path.dirname(os.path.abspath(__file__)))
        with open(src, 'r') as f:
            for line in f:
                stopwords.add(line.rstrip("\r\n"))
        return stopwords
    
def convert_text(text_src, converter, output):
    """Convert a text data to a special vector
    
    """
    if not isinstance(text_src, str):
        raise TypeError("text_src for convert_text type is {}, but it must be string type.".format(type(text_src)))

    if not isinstance(output, str):
        raise TypeError("output for convert_text type is {}, but it must be string type.".format(type(output)))
   
    converter.


    
    
    
        
     
    
    


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
       
    
    


class the specified split_ratio(test_number/total_number)

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





        
    
                    
                        
                
