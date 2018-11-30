# -*- coding: utf-8 -*-
# 加减法练习
import tensorflow as tf
x = tf.Variable([1,2])
a = tf.constant([3,3])
#增加一个减法OP
sub = tf.subtract(x,a)
#增加一个加法OP
add = tf.add(x,sub)

#创建一个变量，初始化为零
state = tf.Variable(0,name='conuter')
#创建一个OP，使得state自增1
new_value = tf.add(state, 1)
#赋值OP
update = tf.assign(state, new_value)

#全局初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    print(sess.run(add))#  op的操作与之前操作的顺序无关  
    print(state)#打印显示state是一个tf的变量
    print(sub) #打印显示sub是一个张量
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
    