import tensorflow as tf
import numpy as np

output = [[[1.0, 0.2, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]]  # 3 * 2 * 3    # batch * 序列长度 * rnn_size
softmax_W = [[1, 2, 3], [5, 6, 7], [9, 10, 11], [0.1, 0.2, 0.3]]   # 4*3    # n_items * rnn_size
softmax_b = [1, 2, 3, 4]
softmax_WT = tf.transpose(softmax_W)
output_shape = tf.shape(output)
softmax_W_re = tf.reshape(tf.tile(softmax_WT, [output_shape[0], 1]), [output_shape[0], tf.shape(softmax_WT)[0], tf.shape(softmax_WT)[1]])
c = tf.matmul(output, softmax_W_re) + softmax_b
softmax = tf.nn.softmax(output)
with tf.Session() as sess:
    print(sess.run(softmax))
    print(softmax)
    # print(sess.run(softmax_W_re))
    # print(softmax_W_re)