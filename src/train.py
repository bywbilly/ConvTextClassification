#coding:utf-8
import os
import sys
import time
import random
import json
import re
import argparse

import numpy as np
import tensorflow as tf

import tfnnutils
from read_data import Data_reader
from feature import Feature

class ConvTextClassfication(object):
    
    def __init__(self, args):
        self.args = args
        self.data_reader = Data_reader()
        self.raw_train_data = self.data_reader.read_train_data()
        #self.raw_test_data = self.data_reader.read_test_data()

        self.feature = Feature(args)
        self.train_data = []
        self.labels = []
        self.test_data = []

    def process_data(self):
        self.train_data, self.labels = self.feature.extract_feature(self.raw_train_data)

    def _loss(self, logits, L2_loss, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='aaa')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='ppp')
        return cross_entropy_mean 

    def _forward(self, batch_x):
        layers = []
        layers.append(tfnnutils.InputLayer())
        layers.append(tfnnutils.Conv2D('conv1', ksize=(self.args.feature, 7), kernels=1)) 
        layers.append(tfnnutils.MaxPool((1, 3)))
        layers.append(tfnnutils.Conv2D('conv2', ksize=(self.args.feature, 7), kernels=1)) 
        layers.append(tfnnutils.MaxPool((1, 3)))
        layers.append(tfnnutils.Conv2D('conv3', ksize=(self.args.feature, 3), kernels=1)) 
        layers.append(tfnnutils.Conv2D('conv4', ksize=(self.args.feature, 3), kernels=1)) 
        layers.append(tfnnutils.Conv2D('conv5', ksize=(self.args.feature, 3), kernels=1)) 
        layers.append(tfnnutils.Conv2D('conv6', ksize=(self.args.feature, 3), kernels=1)) 
        layers.append(tfnnutils.MaxPool((1, 3)))
        layers.append(tfnnutils.Flatten())
        layers.append(tfnnutils.FCLayer('FC1', 1024, act = tf.nn.relu))
        layers.append(tfnnutils.FCLayer('FC2', 1024, act = tf.nn.relu))
        layers.append(tfnnutils.FCLayer('FC3', 2, act = tf.nn.relu))

        L2_loss = 0.
        last_layer = None
        for i, layer in enumerate(layers):
            if hasattr(layer, 'L2_Loss'):
                L2_loss += layer.L2_Loss
            batch_x = layer.forward(last_layer, batch_x)
            last_layer = layer

        pred = tf.nn.softmax(batch_x)

        return pred, batch_x, L2_loss

    def build_model(self):
        global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        self._x = tf.placeholder(tf.float32, shape=[self.args.BatchSize, self.args.feature, self.args.length, 1])
        self._y = tf.placeholder(tf.int32)
        x = self._x
        y = self._y

        pred, logits, L2_loss = self._forward(x)
        loss = self._loss(logits, L2_loss, y)

        grads = opt.compute_gradients(loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer() 

        self.sess = tf.Session()

        self.sess.run(init)

        self.train_step = apply_gradient_op
        self.pred_step  = pred
        self.loss_step  = loss

    def get_batch(self, dataset, index):
        #print 'start getting a batch'
        st = index * self.args.BatchSize
        ed = st + self.args.BatchSize
        #print '---------'
        #print len(dataset)
        if ed >= len(dataset):
            return None, None
        ret_x = np.zeros((self.args.BatchSize, self.args.feature, self.args.length), np.float32)
        ret_y = np.zeros((self.args.BatchSize, ), np.int32)
        #ret_x = np.array(dataset[st:ed])
        #ret_y = np.array(self.labels[st:ed])
        #for i in xrange(st, ed):
        #    print type(dataset[i]['content'])
        #    ret_x[i] = np.array(dataset[i]['content'])
        #    ret_y[i] = np.array(self.labels[i])

        ret_x = ret_x.reshape(self.args.BatchSize, self.args.feature, self.args.length, 1)
       
        return ret_x, ret_y

    def evaluate(self, dataset):
        batch_size = self.args.BatchSize
        total_loss = 0.
        total_err = 0.
        n_batch = 0
        now_pos = 0
        print 'start evaluating'
        while True:
            prepared_x, prepared_y = self.get_batch(dataset, n_batch)
            if prepared_x is None:
                break
            feed = {self._x: prepared_x, self._y: prepared_y}
            loss, preds = self.sess.run([self.loss_step, self.pred_step], feed_dict=feed)
            total_loss += np.mean(loss)
            for i in xrange(len(preds)):
                if np.argmax(preds[i]) != prepared_y[i]:
                    total_err += 1

            n_batch += 1

        loss = total_loss / n_batch
        err = total_err / (n_batch * batch_size)

        print 'evaluate %s: loss = %f err = %f' % (dataset, loss, err)
        
        return loss, err

    def train(self):
        lr = self.args.lr
        best_acc = 0.0
        for epoch in xrange(self.args.num_epoch):
            n_train_batch = 0
            print n_train_batch
            batch_size = self.args.BatchSize
            if epoch > 0  and epoch % 3 == 0:
                lr /= 2.0
            while True:
                prepared_x, prepared_y = self.get_batch(self.train_data, n_train_batch)
                if prepared_x is None:
                    print 'miemiemie'
                    break
                feed = {self.lr: lr, self._x: prepared_x, self._y: prepared_y}
                _, loss = self.sess.run([self.train_step, self.loss_step], feed_dict=feed)
                if n_train_batch % 100 == 0:
                    print 'The iteration is %d train loss is: %f' % (n_train_batch, loss)
                if n_train_batch % 1000 == 0:
                    self.evaluate(self.train_data)

                n_train_batch += 1


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--BatchSize', type=int, default=128)
    argparser.add_argument('--feature', type=int, default=70)
    argparser.add_argument('--length', type=int, default=1018)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--num_epoch', type=int, default=30)

    args = argparser.parse_args()

    model = ConvTextClassfication(args)
    
    print 'start process data'
    model.process_data()
    print 'end prosess data'
    print 'start build model'
    model.build_model()
    print 'end build model'
    print 'start training model'
    model.train()
    print 'finish training model'

    

