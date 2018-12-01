# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:40:03 2018
Huaqiao University 
@author: lopo
"""
import tensorflow as tf
from PIL import Image
from nets import nets_factory
import numpy as np
import matplotlib.pyplot as plt

CHAR_SET_LEN = 10

#批次
BATCH_SIZE = 1
# tf测试集存放文件地址
TFRECORD_FILE = 'E:/tensorflow/lesson/captcha/test.tfrecord'
#定义变量名
x = tf.placeholder(tf.float32,[None,224,224])

# 从tf测试集中提取数据
def read_and_decode(filename):
    tf_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tf_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image':tf.FixedLenFeature([], tf.string),
                                            'label0': tf.FixedLenFeature([], tf.int64),
                                            'label1': tf.FixedLenFeature([], tf.int64),
                                            'label2': tf.FixedLenFeature([], tf.int64),
                                            'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    image = tf.decode_raw(features['image'], tf.uint8)
    #没有处过预处理的灰度图
    image_raw = tf.reshape(image, [224, 224])

    image = tf.reshape(image,[224,224])
    #图片的预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    #获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    return image_raw, image, label0, label1, label2, label3

image_raw, image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)
image_raw_batch, image_batch, label0_batch, label1_batch, label2_batch, label3_batch = tf.train.shuffle_batch(
    [image_raw, image, label0, label1, label2, label3],
    batch_size=BATCH_SIZE,
    capacity=10000,
    min_after_dequeue=2000,
    num_threads=1
)

# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2_captcha_multi',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=False
)

with tf.Session() as sess:
	X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
	logits0, logits1, logits2, logits3, end_points = train_network_fn(X)

	predict0 = tf.reshape(logits0, [-1, CHAR_SET_LEN])
	predict0 = tf.argmax(predict0, 1)
	
	predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])
	predict1 = tf.argmax(predict1, 1)
	
	predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])
	predict2 = tf.argmax(predict2, 1)
	
	predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])
	predict3 = tf.argmax(predict3, 1)	

	sess.run(tf.global_variables_initializer())

	#载入训练好的模型
	saver = tf.train.Saver()
	saver.restore(sess, 'E:/tensorflow/lesson/captcha/')
	
	#创建一个协调器
	coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

	for i in range(10):

		b_image_raw, b_image, b_label0, b_label1, b_label2, b_label3 = sess.run([image_raw_batch,
																	image_batch,
																	label0_batch,
																	label1_batch,
																	label2_batch,
																	label3_batch])
		#显示图片 灰度显示
		img = Image.fromarray(b_image_raw[0],'L')
		plt.imshow(img)
		plt.axis('off')
		plt.show()
		#打印标签
		print('label:', b_label0, b_label1, b_label2, b_label3)
		#预测
		label0, label1, label2, label3 = sess.run([predict0, predict1, predict2, predict3], feed_dict={x: b_image})
		#打印预测值
		print('predict:',label0,label1,label2,label3)
	#通知其它线程关闭
	coord.request_stop()
	coord.join(threads)