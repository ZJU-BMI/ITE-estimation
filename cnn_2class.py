import tensorflow as tf
import numpy as np

class CNN2class(object):
    def __init__(self, dense_units=8, name="cnn", sess=None):

        self.name=name
        self.dense_units = dense_units
        with tf.variable_scope(self.name):

            self.x = tf.placeholder("float", shape=[None, 25])
            self.y = tf.placeholder("float", shape=[None, 1])
            self.t_class = tf.placeholder("float", shape=[None, 1]) #treatment class
            self.tmp = tf.placeholder("float", shape=[None, 1])

            self.input_layer = tf.reshape(self.x, [-1, self.x.shape[1], 1])
            self.conv1 = tf.layers.conv1d(
                inputs=self.input_layer,
                filters=16,
                kernel_size=[8],
                padding="same",
                activation=tf.nn.relu)

            self.pool1 = tf.layers.max_pooling1d(inputs=self.conv1, pool_size=[1], strides=1)
            self.pool1_flat = tf.reshape(self.conv1, [-1, self.conv1.shape[1] * self.conv1.shape[2]])
            self.dense = tf.layers.dense(inputs=self.pool1_flat, units=self.dense_units, activation=tf.nn.sigmoid)
            self.dropout = tf.layers.dropout(inputs=self.dense, rate=0.2)

            #  2 * [N, 1]
            self.logit_0_ = tf.layers.dense(inputs=self.dropout, units=1)
            self.logit_1_ = tf.layers.dense(inputs=self.dropout, units=1)

            self.y_0 = tf.where(tf.equal(self.t_class, 0), self.y, self.tmp)
            self.y_1 = tf.where(tf.equal(self.t_class, 1), self.y, self.tmp)

            self.logit_0 = tf.where(tf.equal(self.t_class, 0), self.logit_0_, self.tmp)
            self.logit_1 = tf.where(tf.equal(self.t_class, 1), self.logit_1_, self.tmp)

            self.loss0 = tf.losses.mean_squared_error(labels=self.y_0, predictions=self.logit_0)
            self.loss1 = tf.losses.mean_squared_error(labels=self.y_1, predictions=self.logit_1)

            self.logits_t = tf.layers.dense(inputs=self.dropout, units=2)
            self.p_t = tf.nn.softmax(self.logits_t)
            # treatment loss
            self.loss_t = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(self.t_class, dtype=tf.int32), logits=self.logits_t)

            self.loss_tol = self.loss0 + self.loss1 + self.loss_t
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            self.train_op = self.optimizer.minimize(
                loss=self.loss_tol,
                global_step=tf.train.get_global_step())

            init = tf.global_variables_initializer()
            c = tf.ConfigProto()
            c.gpu_options.allow_growth = True
            self.sess = sess if sess is not None else tf.Session(config=c)
            self.sess.run(init)

    def train_process(self, dataset):
        tmp = np.zeros((128,1))
        for k in range(5000):
            batch = dataset.next_batch(128)
            x = batch[0]
            y = batch[1][:,0].reshape(-1,1)
            t = batch[1][:,1:9]
            cost, _ = self.sess.run((self.loss_tol, self.train_op), feed_dict={self.x: x, self.y: y, self.t_class: t, self.tmp: tmp})
            if k % 100 == 0:
                print("steps: %d, loss: %f" % (k, cost))

    def cnn_x(self, x):
        cnn_x=self.sess.run(self.dropout, feed_dict={self.x: x})
        return cnn_x

    def predict_0(self, x, t):
        predict_0 = self.sess.run(self.logit_0_, feed_dict={self.x: x, self.t_class: t})
        return predict_0

    def predict_1(self, x, t):
        predict_1 = self.sess.run(self.logit_1_, feed_dict={self.x: x, self.t_class: t})
        return predict_1