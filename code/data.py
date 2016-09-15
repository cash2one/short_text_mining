#-*- coding:utf-8 -*-
import sys
import numpy as np
import collections as coll
import target
import feature
import data
import ConfigParser

class Data(object):

    def __init__(self, data_fin, conf_fin, section, stop_words = None):
        
        #加载stop_words
        if stop_words is not None:
            self.stop_words = self._file2set(fin = stop_words, index = 0)
        else:
            self.stop_words = set()

        #加载配置文件
        configuration = ConfigParser.ConfigParser()
        configuration.read(conf_fin)
        self.config = coll.defaultdict(dict)

        for section in configuration.sections():
            for option in configuration.options(section):
                self.config[section].update({option:configuration.get(section, option)})

        self._load_data(fin = data_fin, section = section)
        
        

   

    def _file2dict(self, fin, index):
        with open(fin, 'r') as f:
            fields_list = [line.strip('\t\n').split('\t')[index] for line in f]
        return dict(zip(list, range(1,len(lfields_list)+1)))

    def _file2set(self, fin, index):
        with open(fin, 'r') as f:
            fields = set([line.strip('\t\n').split('\t')[index] for line in f])
        return fields
    
    def _load_data(self, fin, section):

        #根据配置文件初始化变量
        if "target_index" in self.config[section]:
            target_list = []
        
        #
        if "feat_index" in self.config[section]:
            feature_list = []

        if "title_index" in self.config[section]:
            title_list = []

        number = int(self.config[section]["field_number"])

        #记录加载样本的个数
        sample_num = 0
       
        #加载文件
        with open(fin, "r") as f:
            for line in f:
                fields = line.strip("\t\n").split("\t")
                if len(fields) != number:
                    print("[Data object]-[load_data]-[Warning]: split number: {}, the split number is wrong: {}".format(len(fields), line))
                    continue
                
                sample_num += 1

                if "target_index" in self.config[section]:
                    target_list.append(fields[int(self.config[section]["target_index"])])

                if "feat_index" in self.config[section]:
                    feature_list.append(fields[int(self.config[section]["feat_index"])])

                if "title_index" in self.config[section]:
                    title_list.append(fields[int(self.config[section]["title_index"])])


        if "target_index" in self.config[section] and len(target_list) != sample_num:
            sys.exit("[Data object]-[_load_data]-[Error]: the number of target is not equal to sample_num")
            
        if "feat_index" in self.config[section] and len(feature_list) != sample_num:
            sys.exit("[Data object]-[_load_data]-[Error]: the number of feature is not equal to sample_num. feature:{0}, sample_num:{1}".format(len(feature_list), sample_num))
            
        if "title_index" in self.config[section] and len(title_list) != sample_num:
            sys.exit("[Data object]-[load_train_data]-[Error]: the number of title is not equal to sample_num")

        if sample_num != 0:
            self.total = sample_num

        if "target_index" in self.config[section]:
            self.target = target.Target(target = target_list)

        if "feat_index" in self.config[section]:
            self.feature = feature.Feature(feats = feature_list)

        if "title_index" in self.config[section]:
            self.title = title_list
      
      

    def transformBinary(self, target):
        self.target[self.target != target] = "0"
        self.target[self.target == target] = "1"





        
    
                    
                        
                
