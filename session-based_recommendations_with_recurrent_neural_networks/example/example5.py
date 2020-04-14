import tensorflow as tf
import numpy as np

x = [[[5.0, 0.2, 3.0], [10.0, 5.0, 16.0]], [[7.0, 10.0, 9.0], [12.0, 11.0, 12.0]], [[16.0, 14.0, 15.0], [20.0, 19.0, 18.0]]]  # 3 * 2 * 3    # batch * 序列长度 * n_items
y = tf.Variable(x, name="y")
_, z = tf.nn.top_k(x, 2, sorted=True, name="q")
w = tf.identity(z, name="s")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))
    print(w)