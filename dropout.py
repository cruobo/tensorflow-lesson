# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:40:03 2018
mnist数据集-手写识别简单示例-使用dropout方法
Huaqiao University 
@author: lopo
"""
import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#载入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
#设置批次的大小
batch_size = 100
#计算一共有多少批次(550批次)          ^ //整除操作符
n_batch = mnist.train.num_examples // batch_size 
#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

#创建一个隐含层的神经网络
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
#防止过拟合，使某些训练时失活
L1_drop = tf.nn.dropout(L1,keep_prob)

#创建一个隐含层的神经网络
W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
#防止过拟合，使某些训练时失活
L2_drop = tf.nn.dropout(L2,keep_prob)

#创建一个隐含层的神经网络
W3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
#防止过拟合，使某些训练时失活
L3_drop = tf.nn.dropout(L3,keep_prob)

#定义神经网络输出层
w4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop, w4) + b4)

#二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()
#结果存放在布尔型列表中
#返回最大值所在的索引，并进行比较
correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(41):
        for batch in range(n_batch):
            #获得100张图片
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        test_acc = sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc = sess.run(accuracy,feed_dict = {x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        
        #打印准确率
        print("Iter:"+str(epoch)+";Testing Accuracy:"+str(test_acc)+";Training Accuracy:"+str(train_acc))
            
