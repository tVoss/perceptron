import os
import numpy as np
import pickle

# Hide tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Helper functions
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = len(labels_dense)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def batch_creator(batch_size, data_length, data_name):
    batch_mask = rng.choice(data_length, batch_size)

    batch_x = eval(data_name + '_x')[[batch_mask]].reshape(-1, input_num_units)

    if data_name == 'train':
        batch_y = eval(data_name + '_y')[[batch_mask]]
        batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y

# Load files
print 'Loading data...'
train = pickle.load(open('train.p', 'rb'))
test = pickle.load(open('test.p', 'rb'))

# Gather 'x' data (image values)
train_x = np.array([x[0] for x in train])
train_x, val_x = train_x[:4000], train_x[4000:]
test_x = np.array([x[0] for x in test])

# Gather 'y' data (actual number)
train_y = np.array([y[1] for y in train])
train_y, val_y = train_y[:4000], train_y[4000:]
test_y = np.array([y[1] for y in test])

# Number of nodes for each layer
input_num_units = 784
hidden0_num_units = 1000
hidden1_num_units = 2000
output_num_units = 10

# Describe inputs and outputs
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# Network settings
epochs = 7
batch_size = 300
learning_rate = 0.01

# RNG settings
rng = np.random.RandomState()

# Weights and biases for layers
weights = {
    'hidden0': tf.Variable(tf.random_normal([input_num_units, hidden0_num_units])),
    'hidden1': tf.Variable(tf.random_normal([hidden0_num_units, hidden1_num_units])),
    'output': tf.Variable(tf.random_normal([hidden1_num_units, output_num_units]))
}
biases = {
    'hidden0': tf.Variable(tf.random_normal([hidden0_num_units])),
    'hidden1': tf.Variable(tf.random_normal([hidden1_num_units])),
    'output': tf.Variable(tf.random_normal([output_num_units]))
}

# Link layers
hidden_layer0 = tf.add(tf.matmul(x, weights['hidden0']), biases['hidden0'])
hidden_layer0 = tf.nn.sigmoid(hidden_layer0)
hidden_layer1 = tf.add(tf.matmul(hidden_layer0, weights['hidden1']), biases['hidden1'])
hidden_layer1 = tf.nn.relu(hidden_layer1)
output_layer = tf.matmul(hidden_layer1, weights['output']) + biases['output']

# Create optimization function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize
init = tf.global_variables_initializer()

# Begin network
with tf.Session() as sess:
    sess.run(init)

    # Run as many epochs as requested
    for epoch in xrange(epochs):
        avg_cost = 0
        total_batch = int(len(train) / batch_size)

        # Each epoch is divided into batches
        for i in xrange(total_batch):
            batch_x, batch_y = batch_creator(batch_size, len(train_x), 'train')

            # Train network
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

            avg_cost += c / total_batch

        print 'Epoch:', (epoch + 1), 'cost =', '{:.5f}'.format(avg_cost)

    print 'Training complete!'

    # Test accuracy against training data
    predict = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predict, 'float'))
    print 'Validation Accuracy:', accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)})

    # Test accuracy against test data
    print 'Test Accuracy:', accuracy.eval({x: test_x.reshape(-1, input_num_units), y: dense_to_one_hot(test_y)})
