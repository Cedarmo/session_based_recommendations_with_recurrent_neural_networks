# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class GRU4Rec(object):

    def __init__(self, n_items, rnn_size, learning_rate, init_as_normal,
                 sigma, decay_steps, decay_rate, grad_cap, loss, final_act):
        self.n_items = n_items  # item 总数
        self.rnn_size = rnn_size  # 隐藏层状态
        self.learning_rate = learning_rate  # 学习率
        self.init_as_normal = init_as_normal  # 是否截断初始化, True表示不截断, False表示截断
        self.sigma = sigma  # 初始化标准差
        self.decay_steps = decay_steps  # 衰减速度
        self.decay_rate = decay_rate  # 衰减系数
        self.grad_cap = grad_cap  # 是否对梯度加L2范数, grad_cap 大于 0, 表示加入L2范数，否则不加
        # loss 表示损失函数类型, final_act 表示最终的激活函数
        if loss == 'cross-entropy':
            if final_act == 'softmaxth':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif loss == 'bpr':
            if final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif loss == 'top1':
            if final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError

    ########################ACTIVATION FUNCTIONS#########################
    def tanh(self, X, name):
        return tf.nn.tanh(X, name=name)

    def softmax(self, X, name):
        return tf.nn.softmax(X, name=name)

    def softmaxth(self, X, name):
        return tf.nn.softmax(tf.tanh(X), name=name)

    def relu(self, X, name):
        return tf.nn.relu(X, name=name)

    def sigmoid(self, X, name):
        return tf.nn.sigmoid(X, name=name)

    ############################LOSS FUNCTIONS######################
    def cross_entropy(self, yhat):  # yhat 是 batch * 序列长度 * batch(n_items)
        yhatT = tf.transpose(yhat, [1, 0, 2])
        return tf.reduce_mean(-tf.log(tf.matrix_diag_part(yhatT) + 1e-24), name="loss")

    def bpr(self, yhat):
        yhatT1 = tf.transpose(yhat, [1, 0, 2])  # yhatT1 是 序列长度 * batch * batch(n_items)
        yhatT1_diag = tf.transpose(tf.expand_dims(tf.matrix_diag_part(yhatT1), -1),
                                   [0, 2, 1])  # yhatT1_diag 是 序列长度 * 1 * batch
        yhatT2 = tf.transpose(yhat, [1, 2, 0])  # yhatT2 是 序列长度 * batch(n_items) * batch
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(yhatT1_diag - yhatT2)), name="loss")

    def top1(self, yhat):
        yhatT1 = tf.transpose(yhat, [1, 0, 2])  # yhatT1 是 序列长度 * batch * batch(n_items)
        yhatT1_diag = tf.transpose(tf.expand_dims(tf.matrix_diag_part(yhatT1), -1),
                                   [0, 2, 1])  # yhatT1_diag 是 序列长度 * 1 * batch
        yhatT2 = tf.transpose(yhat, [1, 2, 0])  # yhatT2 是 序列长度 * batch(n_items) * batch
        return tf.reduce_mean(tf.nn.sigmoid(yhatT2 - yhatT1_diag) + tf.nn.sigmoid(yhatT2 ** 2), name="loss")

    # 搭建模型
    def build_model(self):
        self.X = tf.placeholder(tf.int64, [None, None], name='input')  # batch * 序列长度
        self.Y = tf.placeholder(tf.int64, [None, None], name='output')  # batch * 序列长度
        self.seq_len = tf.placeholder(tf.int64, [None], name="seq_len")
        self.drop_out = tf.placeholder(tf.float32, name="drop_out")
        self.topN = tf.placeholder(tf.int32, name="topN")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.batch_size = tf.shape(self.X)[0]

        with tf.variable_scope('gru_layer'):
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + self.rnn_size))
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            partitioner = tf.fixed_size_partitioner(num_shards=self.rnn_size)
            embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size],
                                        tf.float32, initializer=initializer, partitioner=partitioner)
            softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size],
                                        tf.float32, initializer=initializer, partitioner=partitioner)
            softmax_b = tf.get_variable('softmax_b', [self.n_items],
                                        tf.float32, initializer=tf.constant_initializer(0.0), partitioner=partitioner)

            cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.rnn_size)
            initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.drop_out)

            inputs = tf.nn.embedding_lookup(embedding, self.X)  # batch * 序列长度 * rnn_size
            output, state = tf.nn.dynamic_rnn(drop_cell, inputs,
                                              initial_state=initial_state,
                                              dtype=tf.float32,
                                              sequence_length=self.seq_len)
            self.output = output  # batch * 序列长度 * rnn_size
            self.state = state  # batch * rnn_size(最后一个序列的状态)

        '''
        训练
        Use other examples of the minibatch as negative samples.
        '''
        self.sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y, name='sampled_W')  # batch * 序列长度 * rnn_size
        self.sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y, name='sampled_b')  # batch * 序列长度
        # logits_train的shape: batch * 序列长度 * batch
        self.logits_train = tf.transpose(tf.matmul(tf.transpose(output, [1, 0, 2]), tf.transpose(self.sampled_W, [1, 2, 0])),
                              [1, 0, 2]) + tf.expand_dims(self.sampled_b, -1)
        self.yhat_train = self.final_activation(self.logits_train, "yhat_train")  # batch * 序列长度 * batch
        self.cost_train = self.loss_function(self.yhat_train)

        '''
        预测
        '''
        output_shape = tf.shape(output)         # output shape: batch * 序列长度 * rnn_size
        softmax_WT = tf.transpose(softmax_W)        # rnn_size * n_items
        swt_shape = tf.shape(softmax_WT)
        re_softmax = tf.reshape(tf.tile(softmax_WT, [output_shape[0], 1]),
                                [output_shape[0], swt_shape[0], swt_shape[1]])
        self.logits_predict = tf.matmul(output, re_softmax) + softmax_b        # batch * 序列长度 * n_items
        self.yhat_predict = self.final_activation(self.logits_predict, "yhat_predict")
        _, self.indices = tf.nn.top_k(self.yhat_predict, k=self.topN, sorted=True, name="top_k")
        self.index = tf.identity(self.indices, name="index")    # batch * 序列长度 * topN

        '''
        学习率
        '''
        self.lr = tf.maximum(1e-6, tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay_rate, staircase=True))

        '''
        Try different optimizers.
        '''
        # optimizer = tf.train.AdagradOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        # optimizer = tf.train.AdadeltaOptimizer(self.lr)
        # optimizer = tf.train.RMSPropOptimizer(self.lr)

        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost_train, tvars)
        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for (grad, var) in gvs]
        else:
            capped_gvs = gvs
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def parse_fn(self, example):
        "Parse TFExample records."
        example_fmt = {
            'features': tf.VarLenFeature(tf.int64),
            'labels': tf.VarLenFeature(tf.int64),
            'seqLen': tf.FixedLenFeature([], tf.int64)
        }
        parsed = tf.parse_single_example(example, example_fmt)
        return parsed['features'], parsed['labels'], parsed['seqLen']

    def next_batch(self):
        files = tf.data.Dataset.list_files(
            'hdfs路径', shuffle=False
        )
        data_set = files.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=8))
        data_set = data_set.map(map_func=self.parse_fn, num_parallel_calls=8)
        data_set = data_set.prefetch(buffer_size=320)
        data_set = data_set.batch(batch_size=32)
        iterator = data_set.make_one_shot_iterator()
        x, y, seq_len = iterator.get_next()
        return tf.sparse_tensor_to_dense(x), tf.sparse_tensor_to_dense(y), seq_len

def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        #pylint: disable=W0212
        session = session._sess
    return session

def main(_):
    cluster = tf.train.ClusterSpec({"ps": FLAGS.strps_hosts.split(","), "worker": FLAGS.strwork_hosts.split(",")})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                                      cluster=cluster)):
            gru = GRU4Rec(n_items=20000, rnn_size=5, learning_rate=0.0001,
                          init_as_normal=True, sigma=0.1, decay_steps=1e2, decay_rate=0.96,
                          grad_cap=0, loss='cross-entropy', final_act='softmax')
            batch_input, batch_output, batch_seqLen = gru.next_batch()
            gru.build_model()
            hooks = [tf.train.StopAtStepHook(last_step=1002)]
            checkpoint_dir = "hdfs路径"
            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess_config.gpu_options.allow_growth = True
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   checkpoint_dir=checkpoint_dir,
                                                   hooks=hooks,
                                                   config=sess_config,
                                                   save_checkpoint_steps=100) as sess:
                while not sess.should_stop():
                    batch_x, batch_y, seqLen = sess.run([batch_input, batch_output, batch_seqLen])
                    fetches = [gru.output, gru.state, gru.yhat_train, gru.cost_train,
                               gru.yhat_predict, gru.global_step, gru.lr, gru.train_op]
                    feed_dict = {gru.X: batch_x, gru.Y: batch_y, gru.seq_len: seqLen,
                                 gru.drop_out: 0.5, gru.topN: 5}
                    o, s, yt, c, yp, step, _, _ = sess.run(fetches, feed_dict)
                    print("step=", step, "output=", o.shape, "state=", s.shape,
                          "yhatTrain=", yt.shape, "yhatPredict=", yp.shape, "cost=", c)

                # 模型保存
                inputs = {'input': tf.saved_model.utils.build_tensor_info(gru.X),
                          'seq_len': tf.saved_model.utils.build_tensor_info(gru.seq_len),
                          'drop_out': tf.saved_model.utils.build_tensor_info(gru.drop_out),
                          'topN': tf.saved_model.utils.build_tensor_info(gru.topN)}
                outputs = {'index': tf.saved_model.utils.build_tensor_info(gru.index)}
                signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs)
                sess.graph._unsafe_unfinalize()
                builder = tf.saved_model.builder.SavedModelBuilder(
                    "hdfs路径")
                builder.add_meta_graph_and_variables(get_session(sess), [tf.saved_model.tag_constants.SERVING],
                                                     signature_def_map={
                                                         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
                                                     clear_devices=True)
                builder.save()


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string("strps_hosts", "localhost:2000", "参数服务器")
    tf.app.flags.DEFINE_string("strwork_hosts", "localhost:2000", "工作服务器")
    tf.app.flags.DEFINE_string("job_name", "worker", "参数服务器或者工作服务器")
    tf.app.flags.DEFINE_integer("task_index", 0, "job的task索引")
    tf.app.run()