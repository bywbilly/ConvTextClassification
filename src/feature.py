#coding:utf-8
import os
import sys
import time
import random
import json
import re

import numpy as np
from read_data import Data_reader

dict_dir = 'chinese.dict'

class Feature(object):

    def __init__(self, args = None):
        self.data_reader = Data_reader()
        self.args = args
        self.train = []
        self.label = []
        self.dict = {}
        self.__process_dic()

    def __process_dic(self):
        character = None
        with open(dict_dir, 'r') as f:
            character = f.read().split(' ')
        for i in xrange(1, len(character) + 1):
            self.dict[unicode(character[i - 1], 'utf-8')] = i
            #print i

        print character


    def __extract_feature(self, content):
        ret = np.zeros((self.args.feature, self.args.length, 1), float)
        #ret = np.zeros((self.args.feature, self.args.length), float)
        for i, c in enumerate(content):
            if i >= self.args.length:
                break
            if c in self.dict:
                #print c
                if self.dict[c] < 6:
                    ret[self.dict[c]][i][0] = 1.0 
                else:
                    print '--------'
                    print c
                    print self.dict[c]
                    print '---------'
                    break

        return ret

    def extract_feature(self, train_data):
        for i, item in enumerate(train_data):
            print 'I am processing the %d training data' % i
            #feature = train_data[i]
            feature = self.__extract_feature(item['content'])
            label = 0 if (item['label'] == 'true' or item['label'] == 'false') else item['label']
            self.train.append(feature)
            self.label.append(label)

        return self.train, self.label

if __name__ == '__main__':

    feature = Feature()
    test = [{'content': u'我爱吃1，', 'label':  0}] 
    train, label = feature.extract_feature(test)

    #print train
    #print label
        


