# -*- coding: utf-8 -*-
"""
Created on 2018-11-04
mnist数据集-卷积神经网络
Huaqiao University 
@author: lopo
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
# 每一个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 初始化权值
def weight_varibale(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
# 初始化偏置值
def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)
# 卷积层
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
	# x input: tensor of shape [batch, in_height, in_width, in_channels]
	# W filter/kernel tensor of shape, [filter_height, filter_width, in_channels, out_channels]
	# strides[0] = strides[3] = 1, strides[1]: x strides; strides[2]: Y strides
	# padding: A String frome "SAME" or "Valid"

# 池化层
def max_pool_2X2(x):
	# k-size = [1,x,y,1], windows size
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')

# 定义两个placeholder
x = tf.placeholder(tf.float32,[None,784], name = 'x-input')
y = tf.placeholder(tf.float32,[None,10], name = 'y-input')

# 改变X的格式转变为4D的向量[batch,in_height,in_width,channels]
x_image = tf.reshape(x,[-1,28,28,1])

# 初始化第一个卷积层的权值和偏置值
# 5X5 的采样窗口，32个卷积和，从一个平面抽取
W_conv1 = weight_varibale([5,5,1,32])
b_conv1 = bias_variable([32])

# 把 x_image 和权值向量进行卷积，再加上偏置值，然后应用于rulu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2X2(h_conv1)

# 初始化第二个卷积层
# 5X5 的采样窗口，64个卷积核从32个平面抽取特征
W_conv2 = weight_varibale([5,5,32,64])
b_conv2 = bias_variable([64])

# h_pool1 和权值向量进行卷积，再加上偏置值，然后应用于rulu激活函数
# 进行最大值方式的池化
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2X2(h_conv2) 

# 28x28 的图片第一次卷积后还是28x28，第一次池化后变为14x14
# 第二次卷积之后为14x14，第二次池化之后变为7x7
# 通过上面的操作，得到64张7x7的平面图像

# 初始化第一个全连接层的权值和偏置值,该层神经元个数为1024个
W_fc1 = weight_varibale([7*7*64,1024])
b_fc1 = bias_variable([1024])

#把池化层2的输出层扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob 用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第二个全连接层
W_fc2 = weight_varibale([1024,10])
b_fc2 = bias_variable([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用adam优化器进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放于一个列表中
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))#返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(21):
		for batch in range(n_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
		acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
		print("Iter:"+str(epoch)+",Testing Accuracy:"+str(acc))
