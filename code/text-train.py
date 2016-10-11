#-*- coding:utf-8 -*-
import data
import feature
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression








def main(data_fin, feat_num, feat_fout, vectorizer, model, target):
    #数据加载
    data_inst = data.Data()
    data_inst.load_data(fin = data_fin, sep = "\t", key_dict={"title":0,"target":2,"feature":1}, number = 3)
    data_inst.transformBinary(target = target)
    
    #特征选择
    feature_inst = feature.Feature()
    feature_inst.data2vector(data = data_inst, sep = ",")
    feature_inst.select_feat(feat_num = feat_num, algorithm=chi2)
    feature_inst.feat2file(file = feat_fout)
    
    #训练模型
    vocabulary = feature_inst.select_feat_name
    count_vect = CountVectorizer(vocabulary = vocabulary, tokenizer=lambda x: x.split(","))
    X_train = count_vect.fit_transform(data_inst.feature)
    y_train = data_inst.target
    clf = LogisticRegression(C = 100, penalty='l1', tol=0.01)
    clf.fit(X_train, y_train)
    print("sccore with L1 penalty: %.4f" %(clf.score(X_train, y_train))
    



if __name__ == "__main__":

    import sys
    if len(sys.argv) != 7:
        sys.exit("Usage: input feat_num feat_fout vectorizer model target")
        
    data_fin = sys.argv[1]
    feat_num = sys.argv[2]
    feat_fout = sys.argv[3]
    vectorizer = sys.argv[4]
    model = sys.argv[5]
    target = sys.argv[6]
    
    main(data_fin = data_fin, feat_num = feat_num, feat_fout = feat_fout, vectorizer = vectorizer, model = model, target = target)
    
