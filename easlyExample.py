# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:08:55 2018
title: esaly-example using numpy and tensorflow
@author: churuobo
"""
import tensorflow as tf
import numpy as np
#使用numpy生成一百个随机点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2
#构建一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b
#二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
#定义一个使用梯度下降法的优化器,0.2的学习率
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化二次代价函数
train = optimizer.minimize(loss)
#使用到了变量，须对变量进行初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) #初始化
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))

