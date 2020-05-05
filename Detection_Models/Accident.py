import cv2
import tensorflow as tf
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys
from setting import *

class C_DETECTION_YOLO:
    def __init__(self):
        # Network Parameters
        self.n_input = 4096 # fc6 or fc7(1*4096)
        self.n_detection = 20 # number of object of each image (include image features)
        self.n_hidden = 512 # hidden layer num of LSTM
        self.n_img_hidden = 256 # embedding image features
        self.n_att_hidden = 256 # embedding object features
        self.n_classes = 2 # has accident or not
        self.n_frames = 100 # number of frame in each video
        ##################################################

    def detect(self,):
        # build model
        x,keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas = build_model()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        # restore model
        saver.restore(sess, FILE_ADDRESS_DEEP_ACCIDENT_WEIGHT)
    


    def build_model(self):
        # tf Graph input
        x = tf.placeholder("float", [None, self.n_frames ,self.n_detection, n_input])
        y = tf.placeholder("float", [None, self.n_classes])
        keep = tf.placeholder("float",[None])

        # Define weights
        weights = {
            'em_obj': tf.Variable(tf.random_normal([self.n_input,self.n_att_hidden], mean=0.0, stddev=0.01)),
            'em_img': tf.Variable(tf.random_normal([self.n_input,self.n_img_hidden], mean=0.0, stddev=0.01)),
            'att_w': tf.Variable(tf.random_normal([self.n_att_hidden, 1], mean=0.0, stddev=0.01)),
            'att_wa': tf.Variable(tf.random_normal([self.n_hidden, self.n_att_hidden], mean=0.0, stddev=0.01)),
            'att_ua': tf.Variable(tf.random_normal([self.n_att_hidden, self.n_att_hidden], mean=0.0, stddev=0.01)),
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes], mean=0.0, stddev=0.01))
        }
        biases = {
            'em_obj': tf.Variable(tf.random_normal([self.n_att_hidden], mean=0.0, stddev=0.01)),
            'em_img': tf.Variable(tf.random_normal([self.n_img_hidden], mean=0.0, stddev=0.01)),
            'att_ba': tf.Variable(tf.zeros([self.n_att_hidden])),
            'out': tf.Variable(tf.random_normal([self.n_classes], mean=0.0, stddev=0.01))
        }

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden,initializer= tf.random_normal_initializer(mean=0.0,stddev=0.01),use_peepholes = True,state_is_tuple = False)
        # using dropout in output of LSTM
        lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=1 - keep[0])
        # init LSTM parameters
        istate = tf.zeros([self.batch_size, lstm_cell.state_size])
        h_prev = tf.zeros([self.batch_size, self.n_hidden])
        # init loss
        loss = 0.0
        # Mask
        zeros_object = tf.to_float(tf.not_equal(tf.reduce_sum(tf.transpose(x[:,:,1:self.n_detection,:],[1,2,0,3]),3),0)) # frame x n x b
        # Start creat graph
        for i in range(n_frames):
            with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
                # input features (Faster-RCNN fc7)
                X = tf.transpose(x[:,i,:,:], [1, 0, 2])  # permute n_steps and batch_size (n x b x h)
                # frame embedded
                image = tf.matmul(X[0,:,:],weights['em_img']) + biases['em_img'] # 1 x b x h
                # object embedded
                n_object = tf.reshape(X[1:n_detection,:,:], [-1, n_input]) # (n_steps*batch_size, n_input)
                n_object = tf.matmul(n_object, weights['em_obj']) + biases['em_obj'] # (n x b) x h
                n_object = tf.reshape(n_object,[n_detection-1,self.batch_size,self.n_att_hidden]) # n-1 x b x h
                n_object = tf.multiply(n_object,tf.expand_dims(zeros_object[i],2))

                # object attention
                brcst_w = tf.tile(tf.expand_dims(weights['att_w'], 0), [self.n_detection-1,1,1]) # n x h x 1
                image_part = tf.matmul(n_object, tf.tile(tf.expand_dims(weights['att_ua'], 0), [self.n_detection-1,1,1])) + biases['att_ba'] # n x b x h
                e = tf.tanh(tf.matmul(h_prev,weights['att_wa'])+image_part) # n x b x h
                # the probability of each object
                alphas = tf.multiply(tf.nn.softmax(tf.reduce_sum(tf.matmul(e,brcst_w),2),0),zeros_object[i])
                # weighting sum
                attention_list = tf.multiply(tf.expand_dims(alphas,2),n_object)
                attention = tf.reduce_sum(attention_list,0) # b x h
                # concat frame & object
                fusion = tf.concat([image,attention],1)

                with tf.variable_scope("LSTM") as vs:
                    outputs,istate = lstm_cell_dropout(fusion,istate)
                    lstm_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
                # save prev hidden state of LSTM
                h_prev = outputs
                # FC to output
                pred = tf.matmul(outputs,weights['out']) + biases['out'] # b x n_classes
                # save the predict of each time step
                if i == 0:
                    soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(pred),(1,0)),1),(self.batch_size,1))
                    all_alphas = tf.expand_dims(alphas,0)
                else:
                    temp_soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(pred),(1,0)),1),(self.atch_size,1))
                    soft_pred = tf.concat([soft_pred,temp_soft_pred],1)
                    temp_alphas = tf.expand_dims(alphas,0)
                    all_alphas = tf.concat([all_alphas, temp_alphas],0)

                # positive example (exp_loss)
                pos_loss = -tf.multiply(tf.exp(-(self.n_frames-i-1)/20.0),-tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
                # negative example
                neg_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits = pred) # Softmax loss

                temp_loss = tf.reduce_mean(tf.add(tf.multiply(pos_loss,y[:,1]),tf.multiply(neg_loss,y[:,0])))
                #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
                loss = tf.add(loss, temp_loss)

            # Define loss and optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss/self.n_frames) # Adam Optimizer

        return x,keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas