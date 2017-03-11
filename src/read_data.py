#coding:utf-8
import os
import sys
import time
import random
import json
import re

train_dir = '../data/train.json'
test_dir = '../data/test.json'

class Data_reader(object):
    
    def __init__(self):
        self.train_data = []
        self.test_data = []

    def read_train_data(self):
        with open(train_dir, 'r') as f:
            for line in f:
                self.train_data.append(json.loads(line))
        self.erase_html_tags('train')
        return self.train_data

    def read_test_data(self):
        with open(test_dir, 'r') as f:
            for line in f:
                self.test_data.append(json.loads(line))
        self.erase_html_tags('test')
        return self.test_data
    
    def __erase_bracket(self, content):
        #return re.sub("[\(\[].*?[\)\]]", "", content) 
        return re.sub("u【\u（\u【u】.*?u【\u）\u】u】", "") 
        #There are some problems dealing with chinese character, so I write the stupid ones below
        #The stupid version is too slow 
        #ret = ''
        #flip = 0

        #for c in content:
        #    if c == 'u（' or c == 'u）':
        #        flip ^= 1
        #    elif flip == 0:
        #        ret += c
        #    else:
        #        pass

        #return ret
        
    def __erase_html_tags(self, content): 
        html_tags = re.compile('<.*?>')
        content = re.sub(html_tags, '', content)
        return content.replace('\t', '')



    def erase_bracket(self, mode):
        if mode == 'train':
            for i, item in enumerate(self.train_data):
                self.train_data[i]['content'] = self.__erase_bracket(item['content'])
        else:
            for i, item in enumerate(self.test_data):
                self.test_data[i]['content'] = self.__erase_bracket(item['content'])

    def erase_html_tags(self, mode):
        if mode == 'train':
            for i, item in enumerate(self.train_data):
                self.train_data[i]['content'] = self.__erase_html_tags(item['content'])
        else:
            for i, item in enumerate(self.test_data):
                self.test_data[i]['content'] = self.__erase_html_tags(item['content'])



    def test(self):

        for i in xrange(9, 10):
            pass
            #print i
            #print self.test_data[i]['content']

        #self.erase_bracket('test')
        #self.erase_html_tags('test')

        #for i in xrange(9, 10):
            #print i
            #print self.test_data[i]['content']

        #for j in xrange(10):
            #print self.test_data[9]['content'][j]

        #print self.test_data[9]['content'][0] == u'\s'



if __name__ == '__main__':
    data_reader = Data_reader() 
    data_reader.test()




