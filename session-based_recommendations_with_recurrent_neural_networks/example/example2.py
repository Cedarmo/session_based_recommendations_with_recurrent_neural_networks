import tensorflow as tf

with tf.Session() as sess:
    yha = [[[1.0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]], [[5, 0, 0, 0], [0, 6, 0, 0], [0, 0, 7, 0], [0, 0, 0, 8]]]    # 2*4*4
    yhat = tf.transpose(yha, [1, 0, 2])  # 4*2*4
    yhatT1 = tf.transpose(yhat, [1, 0, 2])  # yhatT1 是 序列长度 * batch * batch(n_items)
    yhatT2 = tf.transpose(yhat, [1, 2, 0])  # yhatT2 是 序列长度 * batch(n_items) * batch
    ydia = tf.matrix_diag_part(yhatT1)
    y = tf.reshape(ydia, [ydia.shape[0], 1, ydia.shape[1]]) - yhatT2
    print("yhat=", sess.run(yhat))
    print("yhatT1=", sess.run(yhatT1))
    print("yhatT2=", sess.run(yhatT2))
    print("ydia=", sess.run(ydia))
    print("y=", sess.run(y))