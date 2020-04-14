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

        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                    "hdfs路径")
        signature = meta_graph_def.signature_def
        input_tensor_name = signature[signature_key].inputs['input'].name
        seq_len_tensor_name = signature[signature_key].inputs['seq_len'].name
        drop_out_tensor_name = signature[signature_key].inputs['drop_out'].name
        topN_tensor_name = signature[signature_key].inputs['topN'].name
        index_tensor_name = signature[signature_key].outputs['index'].name

        input = sess.graph.get_tensor_by_name(input_tensor_name)
        seq_len = sess.graph.get_tensor_by_name(seq_len_tensor_name)
        drop_out = sess.graph.get_tensor_by_name(drop_out_tensor_name)
        topN = sess.graph.get_tensor_by_name(topN_tensor_name)
        index = sess.graph.get_tensor_by_name(index_tensor_name)

        # 预测
        for i in range(0, 2):
            x, y, seqL = sess.run([x_batch, y_batch, seq_len_batch])
            feed_dict = {input: x, seq_len: seqL, drop_out: 1.0, topN: 5}
            index_predict = sess.run([index], feed_dict=feed_dict)
            print("index_predict_shape=", index_predict[0].shape)
            print("index_predict=", index_predict)