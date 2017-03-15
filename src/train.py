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
        self.raw_test_data = self.data_reader.read_test_data()

        self.feature = Feature(args)
        self.train_data = []
        self.labels = []
        self.val_data = []
        self.val_labels = []
        self.test_data = []

    def process_data(self):
        self.train_data, self.labels = self.feature.extract_feature(self.raw_train_data)
        self.test_data = self.feature.extract_test_feature(self.raw_test_data)

    def partition_data(self):
        
        num = len(self.train_data)
        partition_point = num - int(num / 10.0)
        #print self.labels

        self.val_data = self.train_data[partition_point:]
        self.val_labels = self.labels[partition_point:]
        self.train_data = self.train_data[:partition_point]
        self.labels = self.labels[:partition_point]


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

        if self.args.test == 1 and self.args.load_model != '':
            print 'Restore the model from %s' % self.args.load_model
            saver = tf.train.Saver()
            saver.restore(self.sess, self.args.load_model)
            print 'Finish restoring the model'


    def get_batch(self, dataset, labels, index):
        #print 'start getting a batch'
        st = index * self.args.BatchSize
        ed = st + self.args.BatchSize
        #print '---------'
        #print len(dataset)
        if ed >= len(dataset):
            return None, None
        ret_x = np.zeros((self.args.BatchSize, self.args.feature, self.args.length), np.float32)
        ret_y = np.zeros((self.args.BatchSize, ), np.int32)
        ret_x = np.array(dataset[st:ed])
        ret_y = np.array(labels[st:ed])
        #for i in xrange(st, ed):
        #    print type(dataset[i]['content'])
        #    ret_x[i] = np.array(dataset[i]['content'])
        #    ret_y[i] = np.array(self.labels[i])

        ret_x = ret_x.reshape(self.args.BatchSize, self.args.feature, self.args.length, 1)
       
        return ret_x, ret_y

    def get_batch_predict(self, dataset, index):
        st = index * self.args.BatchSize
        ed = st + self.args.BatchSize

        if ed >= len(dataset):
            return None

        ret_x = np.zeros((self.args.BatchSize, self.args.feature, self.args.length), np.float32)
        ret_x = np.array(dataset[st:ed])
        ret_y = np.zeros((self.args.BatchSize, ), np.int32)

        ret_x = ret_x.reshape(self.args.BatchSize, self.args.feature, self.args.length, 1)

        return ret_x, ret_y

    def evaluate(self, dataset, labels):
        batch_size = self.args.BatchSize
        total_loss = 0.
        total_err = 0.
        n_batch = 0
        now_pos = 0
        print 'start evaluating'
        while True:
            prepared_x, prepared_y = self.get_batch(dataset, labels, n_batch)
            if prepared_x is None:
                break
            feed = {self._x: prepared_x, self._y: prepared_y}
            loss, preds = self.sess.run([self.loss_step, self.pred_step], feed_dict=feed)
            #print prepared_y[:10]
            #print preds[:10]
            total_loss += np.mean(loss)
            for i in xrange(len(preds)):
                if np.argmax(preds[i]) != prepared_y[i]:
                    total_err += 1

            n_batch += 1
            if n_batch > 10:
                break

        loss = total_loss / n_batch
        err = total_err / (n_batch * batch_size)

        print 'evaluate: loss = %f err = %f' % (loss, err)
        
        return loss, err

    def predict(self, dataset):
        batch_size = self.args.BatchSize
        predictions = []
        n_batch = 0
        now_pos = 0
        print 'starting predicting the test dataset'
        while True:
            prepared_x, prepared_y = self.get_batch_predict(dataset, n_batch)
            if prepared_x is None:
                break
            feed = {self._x: prepared_x, self._y: prepared_y}
            _, preds = self.sess.run([self.loss_step, self.pred_step], feed_dict=feed)
            predictions.extend(preds)

            n_batch += 1

        return predictions



    def save(self, dirname):
        try:
            os.makedirs(dirname)
        except:
            pass
        saver = tf.train.Saver()
        return saver.save(self.sess, os.path.join(dirname, "model1.ckpt"))
   
    def test(self):
        print 'starting test'
        predictions = self.predict(self.test_data)
        with open('ans', 'w') as f:
            for item in predictions:
                try:
                    f.write(item[1])
                except:
                    print item

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
                prepared_x, prepared_y = self.get_batch(self.train_data, self.labels, n_train_batch)
                if prepared_x is None:
                    print 'miemiemie'
                    break
                feed = {self.lr: lr, self._x: prepared_x, self._y: prepared_y}
                _, loss = self.sess.run([self.train_step, self.loss_step], feed_dict=feed)
                if n_train_batch % 100 == 0:
                    print 'The iteration is %d train loss is: %f' % (n_train_batch, loss)
                if n_train_batch % 1000 == 0:
                    self.evaluate(self.val_data, self.val_labels)

                n_train_batch += 1
        print 'start saving the model'
        self.save(args.save_model)
        print 'finish saving the model'


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--BatchSize', type=int, default=128)
    argparser.add_argument('--feature', type=int, default=70)
    argparser.add_argument('--length', type=int, default=1018)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--num_epoch', type=int, default=30)
    argparser.add_argument('--load_model', type=str, default='')
    argparser.add_argument('--save_model', type=str, default='')
    argparser.add_argument('--test', type=int, default=0)
    

    args = argparser.parse_args()

    model = ConvTextClassfication(args)
    
    print 'start process data'
    model.process_data()
    print 'end prosess data'
    print 'strat partition data'
    model.partition_data()
    print 'end partition data'
    print 'start build model'
    model.build_model()
    print 'end build model'
    print 'start training model'
    if args.test == 0:
        model.train()
    else:
        model.test()
    print 'finish training model'

    

