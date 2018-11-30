# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:33:46 2018
fetch and feed 
@author: lopo
"""
import tensorflow as tf
#fetch 即可以执行多个op
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    result = sess.run([mul,add])
    print(result)

#feed
#创建占位符
input4 = tf.placeholder(tf.float32)
input5 = tf.placeholder(tf.float32)
output = tf.multiply(input4, input5)

with tf.Session() as sess:
    #feed的数据以字典的形式传入
    print(sess.run(output, feed_dict={input4:[8.],input5:[2.]}))
