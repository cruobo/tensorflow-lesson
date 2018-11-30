import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)


#设置批次的大小
batch_size = 100
#计算一共有多少批次(550批次)
n_batch = mnist.train.num_examples // batch_size 
#命名空间
with tf.name_scope('input'):
	#定义两个placeholder
	x = tf.placeholder(tf.float32,[None,784], name = 'x-input')
	y = tf.placeholder(tf.float32,[None,10], name = 'y-input')

with tf.name_scope('layer'):
	with tf.name_scope('weights'):
		W = tf.Variable(tf.zeros([784,10]), name='W')
	with tf.name_scope('biases'):
		b = tf.Variable(tf.zeros([10]), name = 'b')
	with tf.name_scope('wx_plus_b'):
		wx_plus_b = tf.matmul(x, W) + b
	with tf.name_scope('softmax'):
		prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
#二次代价函数
	loss = tf.reduce_mean(tf.square(y - prediction))
#使用梯度下降法
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()
#结果存放在布尔型列表中
#返回最大值所在的索引，并进行比较
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
#求准确率
	with tf.name_scope('arruracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            #获得100张图片
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels})
        #打印准确率
        print("Iter:"+str(epoch)+";Testing Accuracy:"+str(acc))