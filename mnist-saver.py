# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:40:03 2018
mnist数据集-手写识别简单示例
华侨大学
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
#计算一共有多少批次(550批次)
n_batch = mnist.train.num_examples // batch_size 
#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,56]))
b = tf.Variable(tf.zeros([56]))

L1 = tf.nn.tanh(tf.matmul(x, W) + b)
#定义神经网络输出层
Weight_L2 = tf.Variable(tf.random_normal([56,10]))
biases_L2 = tf.Variable(tf.zeros([10]))
Wx_plus_b_L2 = tf.matmul(L1, Weight_L2) + biases_L2

prediction = tf.nn.softmax(Wx_plus_b_L2)

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

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            #获得100张图片
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels})
        #打印准确率
        print("Iter:"+str(epoch)+";Testing Accuracy:"+str(acc))
    #保存模型
    saver.save(sess, 'net/my_net.ckpt')