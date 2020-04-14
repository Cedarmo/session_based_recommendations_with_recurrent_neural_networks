import math
import numpy as np
import tensorflow as tf

bool = tf.placeholder(tf.bool)
a = tf.Variable([[1.0, 2], [3, 4], [5, 6]])
b = tf.Variable([[1.0, 2], [4, 5]])
c = tf.Variable([[1, 2.0, 3], [4, 5, 6]])
def fn1(): return tf.matmul(a, b)
def fn2(): return tf.matmul(a, c)
f = tf.cond(bool, lambda:fn1(), lambda:fn2())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {bool: True}
    y = sess.run([f], feed_dict=feed_dict)
    print(y[0])
