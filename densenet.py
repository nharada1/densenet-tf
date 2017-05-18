import numpy as np
import tensorflow as tf

k = 12
depth = 40
N = int((depth-4) // 3)

def _conv2d(input, in_feats, out_feats, kernel_size, name="conv"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, in_feats, out_feats], stddev=np.sqrt(2.0/9/in_feats)), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[out_feats]), name="B")
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
        tf.summary.histogram("weights", W)
        tf.summary.histogram("bias", b)
    return conv + b

def _dense(input, out_feats, name="dense"):
    with tf.name_scope(name):
        in_feats = input.get_shape().as_list()[-1]
        W = tf.Variable(tf.truncated_normal([in_feats, out_feats], stddev=0.01), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[out_feats]), name="b")
        flat = tf.reshape(input, [-1, in_feats])
        tf.summary.histogram("weights", W)
        tf.summary.histogram("bias", b)
    return tf.matmul(flat, W)

def _Hl(name, input, kernel_size, is_training, drop_prob, prev):
    in_feats = input.get_shape().as_list()[3]
    with tf.variable_scope(name) as scope:
        out = tf.contrib.layers.batch_norm(input, scale=True, is_training=is_training, updates_collections=None)
        out = tf.nn.relu(out)
        out = _conv2d(out, in_feats, k, kernel_size, name=name+"_conv")
        out = tf.nn.dropout(out, keep_prob=1-drop_prob)
        # Since the last H layer will have the all the feature maps from previous layers, we can
        # concatenate just the previous one to get the dense connections for free.
        out = tf.concat([out, prev], 3)
    return out

def _Tl(name, input, is_training, drop_prob):
    in_feats = input.get_shape().as_list()[3]
    with tf.variable_scope(name) as scope:
        out = tf.contrib.layers.batch_norm(input, scale=True, is_training=is_training, updates_collections=None)
        out = tf.nn.relu(out)
        out = _conv2d(out, in_feats, in_feats, 1, name=name+"_conv")
        out = tf.nn.dropout(out, keep_prob=1-drop_prob)
        out = tf.nn.avg_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    return out


def inference(images, is_training):
    """Build our Densenet model"""

    with tf.variable_scope('input') as scope:
        out = _conv2d(images, 3, 16, 3)

    # Block 1
    with tf.variable_scope('block1') as scope:
        for i in range(N):
            out = _Hl("dense_{}".format(i), out, 3, is_training=is_training, drop_prob=0.2, prev=out)
        out = _Tl("trans_{}".format(i), out, is_training=is_training, drop_prob=0.2)

    # Block 2
    with tf.variable_scope('block2') as scope:
        for i in range(N):
            out = _Hl("dense_{}".format(i), out, 3, is_training=is_training, drop_prob=0.2, prev=out)
        out = _Tl("trans_{}".format(i), out, is_training=is_training, drop_prob=0.2)

    # Block 3
    with tf.variable_scope('block3') as scope:
        for i in range(N):
            out = _Hl("dense_{}".format(i), out, 3, is_training=is_training, drop_prob=0.2, prev=out)
        out = _Tl("trans_{}".format(i), out, is_training=is_training, drop_prob=0.2)

    with tf.variable_scope('final') as scope:
        out = tf.contrib.layers.batch_norm(out, scale=True, is_training=is_training, updates_collections=None)
        out = tf.nn.relu(out)
        # out = tf.nn.avg_pool(out, [1, 4, 4, 1], [1, 1, 1, 1], 'SAME')
        out = tf.reduce_mean(out, [1, 2], name='GAP')
        logits = _dense(out, 10)
    return logits


def loss(logits, labels):
    """Use standard cross entropy loss"""
    with tf.name_scope("loss"):
        labels = tf.to_int64(labels)
        x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy'
        )
        loss = tf.reduce_mean(x_ent, name='cross_entropy_mean')
        tf.summary.scalar("loss", loss)
    return loss


def training(loss, learning_rate):
    """Train using SGD with Nesterov momentum"""
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(loss)
    return train_op


def evaluation(preds, labels):
    """Take top-1 error"""
    with tf.name_scope("evaluation"):
        correct = tf.cast(tf.nn.in_top_k(preds, labels, 1), tf.float32)
        acc = tf.reduce_mean(correct)

        tf.summary.scalar("eval", acc)
    return acc
