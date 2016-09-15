#-*- coding:utf-8 -*-

def token_split(x):
    return x.split(",")


def list2file(fout, res_list):
    with open(fout, 'w') as f:
        for res in res_list:
            f.write("{}\n".format(res))
