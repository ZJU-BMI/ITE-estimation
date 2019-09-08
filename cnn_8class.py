import tensorflow as tf
import numpy as np

class CNN8class(object):
    def __init__(self, dense_units=8, name="cnn", sess=None):

        self.name=name
        self.dense_units = dense_units
        with tf.variable_scope(self.name):

            self.x = tf.placeholder("float", shape=[None, 105])
            self.y = tf.placeholder("float", shape=[None, 1])
            self.t_class = tf.placeholder("float", shape=[None, 1]) #treatment class
            self.tmp = tf.placeholder("float", shape=[None, 1])

            self.input_layer = tf.reshape(self.x, [-1, self.x.shape[1], 1])
            self.conv1 = tf.layers.conv1d(
                inputs=self.input_layer,
                filters=32,
                kernel_size=[8],
                padding="same",
                activation=tf.nn.relu)

            self.pool1 = tf.layers.max_pooling1d(inputs=self.conv1, pool_size=[1], strides=1)
            self.pool1_flat = tf.reshape(self.conv1, [-1, self.conv1.shape[1] * self.conv1.shape[2]])
            self.dense = tf.layers.dense(inputs=self.pool1_flat, units=self.dense_units, activation=tf.nn.sigmoid) # fully connected layer
            self.dropout = tf.layers.dropout(inputs=self.dense, rate=0.3)

            #  8 * [N, 1]
            self.logit_0_ = tf.layers.dense(inputs=self.dropout, units=1)
            self.logit_1_ = tf.layers.dense(inputs=self.dropout, units=1)
            self.logit_2_ = tf.layers.dense(inputs=self.dropout, units=1)
            self.logit_3_ = tf.layers.dense(inputs=self.dropout, units=1)
            self.logit_4_ = tf.layers.dense(inputs=self.dropout, units=1)
            self.logit_5_ = tf.layers.dense(inputs=self.dropout, units=1)
            self.logit_6_ = tf.layers.dense(inputs=self.dropout, units=1)
            self.logit_7_ = tf.layers.dense(inputs=self.dropout, units=1)

            # sigmoid layers for outcome prediction
            self.pred_class_0 = tf.nn.sigmoid(self.logit_0_)
            self.pred_class_1 = tf.nn.sigmoid(self.logit_1_)
            self.pred_class_2 = tf.nn.sigmoid(self.logit_2_)
            self.pred_class_3 = tf.nn.sigmoid(self.logit_3_)
            self.pred_class_4 = tf.nn.sigmoid(self.logit_4_)
            self.pred_class_5 = tf.nn.sigmoid(self.logit_5_)
            self.pred_class_6 = tf.nn.sigmoid(self.logit_6_)
            self.pred_class_7 = tf.nn.sigmoid(self.logit_7_)

            self.y_0 = tf.where(tf.equal(self.t_class, 0), self.y, self.tmp)
            self.y_1 = tf.where(tf.equal(self.t_class, 1), self.y, self.tmp)
            self.y_2 = tf.where(tf.equal(self.t_class, 2), self.y, self.tmp)
            self.y_3 = tf.where(tf.equal(self.t_class, 3), self.y, self.tmp)
            self.y_4 = tf.where(tf.equal(self.t_class, 4), self.y, self.tmp)
            self.y_5 = tf.where(tf.equal(self.t_class, 5), self.y, self.tmp)
            self.y_6 = tf.where(tf.equal(self.t_class, 6), self.y, self.tmp)
            self.y_7 = tf.where(tf.equal(self.t_class, 7), self.y, self.tmp)

            self.logit_0 = tf.where(tf.equal(self.t_class, 0), self.logit_0_, self.tmp)
            self.logit_1 = tf.where(tf.equal(self.t_class, 1), self.logit_1_, self.tmp)
            self.logit_2 = tf.where(tf.equal(self.t_class, 2), self.logit_2_, self.tmp)
            self.logit_3 = tf.where(tf.equal(self.t_class, 3), self.logit_3_, self.tmp)
            self.logit_4 = tf.where(tf.equal(self.t_class, 4), self.logit_4_, self.tmp)
            self.logit_5 = tf.where(tf.equal(self.t_class, 5), self.logit_5_, self.tmp)
            self.logit_6 = tf.where(tf.equal(self.t_class, 6), self.logit_6_, self.tmp)
            self.logit_7 = tf.where(tf.equal(self.t_class, 7), self.logit_7_, self.tmp)

            self.loss0 = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y_0, logits=self.logit_0)
            self.loss1 = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y_1, logits=self.logit_1)
            self.loss2 = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y_2, logits=self.logit_2)
            self.loss3 = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y_3, logits=self.logit_3)
            self.loss4 = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y_4, logits=self.logit_4)
            self.loss5 = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y_5, logits=self.logit_5)
            self.loss6 = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y_6, logits=self.logit_6)
            self.loss7 = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y_7, logits=self.logit_7)



            self.logits_t = tf.layers.dense(inputs=self.dropout, units=8)
            # softmax layer for treatment classification
            self.p_t = tf.nn.softmax(self.logits_t)

            # treatment classification loss
            self.loss_t = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(self.t_class, dtype=tf.int32), logits=self.logits_t)
            # total loss
            self.loss_tol = self.loss0 + self.loss1 + self.loss2 + self.loss3 + self.loss4 + self.loss5 + self.loss6 + self.loss7 + self.loss_t

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
            if k % 1000 == 0:
                print("steps: %d, loss: %f" % (k, cost))

    def cnn_x(self, x):
        cnn_x=self.sess.run(self.dropout, feed_dict={self.x: x})
        return cnn_x

    def predict_0(self, x, t):
        predict_0 = self.sess.run(self.pred_class_0, feed_dict={self.x: x, self.t_class: t})
        return predict_0

    def predict_1(self, x, t):
        predict_1 = self.sess.run(self.pred_class_1, feed_dict={self.x: x, self.t_class: t})
        return predict_1

    def predict_2(self, x, t):
        predict_0 = self.sess.run(self.pred_class_2, feed_dict={self.x: x, self.t_class: t})
        return predict_0

    def predict_3(self, x, t):
        predict_1 = self.sess.run(self.pred_class_3, feed_dict={self.x: x, self.t_class: t})
        return predict_1

    def predict_4(self, x, t):
        predict_0 = self.sess.run(self.pred_class_4, feed_dict={self.x: x, self.t_class: t})
        return predict_0

    def predict_5(self, x, t):
        predict_1 = self.sess.run(self.pred_class_5, feed_dict={self.x: x, self.t_class: t})
        return predict_1

    def predict_6(self, x, t):
        predict_0 = self.sess.run(self.pred_class_6, feed_dict={self.x: x, self.t_class: t})
        return predict_0

    def predict_7(self, x, t):
        predict_7 = self.sess.run(self.pred_class_7, feed_dict={self.x: x, self.t_class: t})
        return predict_7
