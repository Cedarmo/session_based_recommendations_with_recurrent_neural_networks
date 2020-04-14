import math
import numpy as np
import tensorflow as tf

class dataProcess(object):

    def parse_fn(self, example):
        "Parse TFExample records."
        example_fmt = {
            'features': tf.VarLenFeature(tf.int64),
            'labels': tf.VarLenFeature(tf.int64),
            'seqLen': tf.FixedLenFeature([], tf.int64)
        }
        parsed = tf.parse_single_example(example, example_fmt)
        return parsed['features'], parsed['labels'], parsed['seqLen']

    def next_batch(self, batch_size):
        files = tf.data.Dataset.list_files(
            'hdfs路径', shuffle=False
        )
        data_set = files.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=16))
        data_set = data_set.map(map_func=self.parse_fn, num_parallel_calls=16)
        data_set = data_set.prefetch(buffer_size=640)
        data_set = data_set.batch(batch_size=batch_size)
        iterator = data_set.make_one_shot_iterator()
        x, y, seq_len = iterator.get_next()
        return tf.sparse_tensor_to_dense(x), tf.sparse_tensor_to_dense(y), seq_len


if __name__ == "__main__":
    with tf.Session() as sess:
        # 数据预处理#
        dataProcess = dataProcess()
        x_batch, y_batch, seq_len_batch = dataProcess.next_batch(batch_size=2)

        # 加载模型
        saver = tf.train.import_meta_graph("hdfs路径", clear_devices=True)
        saver.restore(sess, "hdfs路径")
        graph = tf.get_default_graph()
        x_input = graph.get_operation_by_name("input").outputs[0]
        y_actual = graph.get_operation_by_name("output").outputs[0]
        seqL = graph.get_operation_by_name("seq_len").outputs[0]
        drop_out = graph.get_operation_by_name("drop_out").outputs[0]

        # 预测
        for i in range(0, 2):
            x, y, seq_len = sess.run([x_batch, y_batch, seq_len_batch])
            feed_dict_test_data = {x_input: x, y_actual: y, seqL: seq_len, drop_out: 1.0}
            yhat_predict = sess.run([graph.get_operation_by_name("yhat_predict").outputs[0]],
                                            feed_dict=feed_dict_test_data)
            yhat_predict_shape = yhat_predict[0].shape
            print("yhat_predict_shape=", yhat_predict_shape)
            print("yhat_predict=", yhat_predict)