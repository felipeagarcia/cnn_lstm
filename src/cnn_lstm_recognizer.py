import tensorflow as tf
from tensorflow.contrib import rnn
import data_handler as data
import numpy as np

# data format: 150 x 19
hm_epochs = 1000
n_classes = 19
batch_size = 8
chunk_size = 19
n_chunks = 153
rnn_size = 256
max_len = 153

x = tf.placeholder(tf.float32, [None, n_chunks, chunk_size])
y = tf.placeholder('float')

# keep_rate = 0.8
# keep_prob = tf.placeholder(tf.float32)

inputs, labels = data.open_data(max_len=max_len)
inputs, labels = np.array(inputs), np.array(labels)
print(len(inputs), len(labels))
test_inputs = inputs[int(0.8*len(inputs)):len(inputs)]
test_labels = labels[int(0.8*len(labels)):len(labels)]
inputs = inputs[0:int(0.8*len(inputs))]
labels = labels[0:int(0.8*len(labels))]
lengs = [len(x) for x in inputs]
print(lengs, max(lengs))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'out': tf.Variable(tf.random_normal([rnn_size, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'out': tf.Variable(tf.random_normal([n_classes]))}
    aux = x
    x = tf.reshape(x, [-1, 150, 19, 1])
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    x = aux
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # fc = tf.nn.dropout(fc, keep_rate)

    output = tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,
                                                       labels=y)
            )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(inputs):
                epoch_x = np.array(inputs[i:i+batch_size])
                epoch_y = np.array(labels[i:i+batch_size])
                _, c = sess.run([optimizer, cost],
                                feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch, 'completed out of',
                  hm_epochs, 'loss:', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: np.array(test_inputs),
              y: np.array(test_labels)}))
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)


train_neural_network(x)
