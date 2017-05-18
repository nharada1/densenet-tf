import argparse
import pickle
import os

import numpy as np
import tensorflow as tf

import densenet


def load_data(folder):
    """Loads the CIFAR data into test and train datasets"""
    def load_pickle(path):
        with open(path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        return d

    def normalize(img, testimg):
        means = img.mean(axis=0)
        stds = img.std(axis=0)

        train = ((img-means)/stds)
        test = ((testimg-means)/stds)
        return train, test


    # X, y = [], []
    # for x in range(1, 6):
    #     path = os.path.join(folder, "data_batch_{}".format(x))
    #     d = load_pickle(path)
    #     X.append(d[b'data'])
    #     y.append(d[b'labels'])
    #
    # X_train = np.concatenate(X)
    # y_train = np.concatenate(y)
    #
    # d_t = load_pickle(os.path.join(folder, 'test_batch'))
    # X_test = d_t[b'data']
    # y_test = d_t[b'labels']

    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    y_train, y_test = y_train.flatten(), y_test.flatten()
    # Channel mean and std dev normalization
    X_train, X_test = normalize(X_train, X_test)

    return X_train, X_test, y_train, y_test


def main(args):
    X_train, X_test, y_train, y_test = load_data(args.cifar_python_folder)
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(None))
        is_training = tf.placeholder("bool", shape=[])
        learning_rate = tf.placeholder("float", shape=[])

        logits = densenet.inference(images_placeholder, is_training)
        loss = densenet.loss(logits, labels_placeholder)
        train_op = densenet.training(loss, learning_rate=learning_rate)
        correct = densenet.evaluation(logits, labels_placeholder)

        init = tf.global_variables_initializer()
        sess = tf.Session()

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('checkpoints/'+args.name)
        writer.add_graph(sess.graph)

        sess.run(init)

        summ = tf.summary.merge_all()
        epoch, idx, lr = 0, 0, 0.1
        for step in range(args.max_steps):
            # Check if this is the last of the examples and reset for the next epoch
            curlen = len(y_train[idx:idx+args.batch_size])
            if curlen == 0:
                idx = 0
                epoch += 1
                print("Epoch movement, now on EPOCH {}".format(epoch))

                acc = []
                i = 0
                for x in range(int(len(X_test)/args.batch_size)):
                    X = X_test[i:i+args.batch_size, :, :, :]
                    y = y_test[i:i+args.batch_size]
                    feed_dict = {
                        images_placeholder: X,
                        labels_placeholder: y,
                        is_training: False,
                        learning_rate: lr
                    }
                    i += args.batch_size
                    [test_acc, s] = sess.run([correct, summ], feed_dict=feed_dict)
                    writer.add_summary(s, step)
                    acc.append(test_acc)
                print('Validation accuracy = {:.4%}'.format(np.mean(acc)))


            X = X_train[idx:idx+args.batch_size, :, :, :]
            y = y_train[idx:idx+args.batch_size]

            feed_dict = {
                images_placeholder: X,
                labels_placeholder: y,
                is_training: True,
                learning_rate: lr
            }
            sess.run(train_op, feed_dict=feed_dict)

            idx += len(y)

            # Metrics
            if step % 200 == 0:
                [loss_scalar, s] = sess.run([loss, summ], feed_dict=feed_dict)
                writer.add_summary(s, step)
                print('Step {0}: loss = {1:.4f}'.format(step, loss_scalar))


            # Learning rate drop
            if epoch == 150:
                lr = 0.01
            if epoch == 225:
                lr = 0.001


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cifar_python_folder', help='Location of unzipped CIFAR python data')
    parser.add_argument('name', help='Name of this run')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-l', '--base_lr', type=float, default=0.1, help='Base learning rate')
    parser.add_argument('-s', '--max_steps', type=int, default=35000, help='Max training steps')
    args = parser.parse_args()
    main(args)