#coding:utf-8
#import the tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#define the functions of W,b,conv2d,pool
def weight_variable(shape):
	weight = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(weight)

def bias_variable(shape):
	bias = tf.constant(0.1, shape=shape)
	return tf.Variable(bias)

def conv2d(x,W):
	conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
	return conv

def pool_max_2x2(x):
	pooling = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	return pooling

#here is the main function
if __name__ == '__main__':
	#have the data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
	
	#have the placeholder
	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])

	#transform the x_array to x_image
	x_image = tf.reshape(x, [-1,28,28,1])

	#the first layer
	W_conv1 = weight_variable([5,5,1,32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	pool_conv1 = pool_max_2x2(h_conv1)

	#the second layer 
	W_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(pool_conv1, W_conv2) + b_conv2)
	pool_conv2 = pool_max_2x2(h_conv2)

	#the fullly connected layer
	W_fcv1 = weight_variable([7*7*64, 1024])
	b_fcv1 = bias_variable([1024])
	h_flat1 = tf.reshape(pool_conv2, [-1, 7*7*64])
	h_flcv1 = tf.nn.relu(tf.matmul(h_flat1, W_fcv1) + b_fcv1)

	#define the keep_prob
	keep_prob = tf.placeholder(tf.float32)
	h_flcv1_drop = tf.nn.dropout(h_flcv1, keep_prob)

	#dimensionality reduction
	W_fcv2 = weight_variable([1024,10])
	b_fcv2 = bias_variable([10])
	h_flcv2 = tf.matmul(h_flcv1_drop, W_fcv2) + b_fcv2

	#cross_entropy 
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_flcv2))

	#train_step 
	train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

	#define the correct_prediction and accuracy
	correct_prediction = tf.equal(tf.argmax(h_flcv2, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#create the session
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())


	#train 2000
	for i in range(2000):
		#have the batch
		batch = mnist.train.next_batch(50)
		#each 100 step print the accuracy
		train_accuracy = sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                #check it each 100 steps
		if i%100 == 0:
			test_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
			print('step:%d,accuracy: %g'%(i,test_accuracy))


	print('the accuracy %g'%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
