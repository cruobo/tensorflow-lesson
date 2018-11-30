# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 00:03:14 2018
@author: lopo
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
 
tf.set_random_seed(1)
np.random.seed(1)
#构建所需数据
x = np.linspace(-1, 1, 100)[:, np.newaxis]         
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                     
#输入x和y
tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.float32, y.shape)
# 搭建神经网络
#隐藏层
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
#输出层
output = tf.layers.dense(l1, 1)
loss = tf.losses.mean_squared_error(tf_y, output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
plt.ion()
#训练
for step in range(500):
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 5 == 0:
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)  
plt.ioff()
plt.show()
