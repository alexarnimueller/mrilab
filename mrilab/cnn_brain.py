"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy
from mrilab.brain import MultiBrain
from mrilab.preprocessing import fnames_to_targets
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches.
    Small utility function to evaluate a dataset by feeding batches of data to
    {eval_data} and pulling the results from {eval_predictions}.
    Saves memory and enables this to run on smaller GPUs.
    """
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run( eval_prediction, feed_dict={eval_data: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(eval_prediction, feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

# variables
outfile = '/Users/modlab/Desktop/predictions.csv'
dir_train = '/Users/modlab/Desktop/Alex/data/set_train'
dir_test = '/Users/modlab/Desktop/Alex/data/set_test'
IMAGE_SIZE = 56
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 100
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 100
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

# def main(_):
# Extract training brains into arrays
datadict_train = fnames_to_targets(dir_train)
datadict_test = fnames_to_targets(dir_test)
train = MultiBrain(datadict_train, dir_train, sort=False)
#test = MultiBrain(datadict_test, dir_test)
train.combine_brains()
#test.combine_brains()
train.cube_mask(x1=4, x2=172, y1=20, y2=188, z1=4, z2=172, apply=True, plot=False)  # cut to cube
#test.cube_mask(x1=4, x2=172, y1=20, y2=188, z1=4, z2=172, apply=True, plot=False)  # cut to cube
#    test.cube_mask(x1=20, x2=150, y1=20, y2=180, z1=20, z2=150, apply=True, plot=False)  # cut out empty slices
train.data_to_img((168, 168, 168))
#test.data_to_img((168, 168, 168))
train.vox_brain((3, 3, 3), plot=False)
#test.vox_brain((3, 3, 3), plot=False)
train.rescale(nbins=256, numproc=12)
#test.rescale(nbins=256, numproc=12)

# Reshape horizontal image slices into one big image array with corresponding labels
train.stack_all_slices()
#test.stack_all_slices()
train.data = train.data.reshape(tuple([1] + list(train.data.shape)))
#test.data = test.data.reshape(tuple([1] + list(test.data.shape)))

# Generate a validation set.
validation_data = train.data.T[:VALIDATION_SIZE, ...]
validation_labels = train.targets[:VALIDATION_SIZE]
train_data = train.data.T[VALIDATION_SIZE:, ...]
train_labels = train.targets[VALIDATION_SIZE:]
num_epochs = NUM_EPOCHS
train_size = len(train_labels)

# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.
train_data_node = tf.placeholder('float32', shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
eval_data = tf.placeholder('float32', shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

# The variables below hold all the trainable weights. They are passed an
# initial value which will be assigned when we call:
# {tf.initialize_all_variables().run()}         # 5x5 filter, depth 32.
conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype='float32'))
conv1_biases = tf.Variable(tf.zeros([32], dtype='float32'))
conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype='float32'))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype='float32'))
fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev=0.1,
                                              seed=SEED, dtype='float32'))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype='float32'))
fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED, dtype='float32'))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype='float32'))

# Training computation: logits + cross-entropy loss.
logits = model(train_data_node, train=True)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_labels_node))

# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
# Add the regularization term to the loss.
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0, dtype='float32')

# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True)

# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

# Predictions for the current training minibatch.
train_prediction = tf.nn.softmax(logits)

# Predictions for the test and validation, which we'll compute less often.
eval_prediction = tf.nn.softmax(model(eval_data))

# Create a local session to run the training.
start_time = time.time()
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print('Initialized!')
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
        # Run the optimizer to update weights.
        sess.run(optimizer, feed_dict=feed_dict)
        # print some extra information once reach the evaluation frequency
        if step % EVAL_FREQUENCY == 0:
            # fetch some extra nodes' data
            l, lr, predictions = sess.run([loss, learning_rate, train_prediction], feed_dict=feed_dict)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * BATCH_SIZE / train_size,
                   1000 * elapsed_time / EVAL_FREQUENCY))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
            print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
            print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
            sys.stdout.flush()
