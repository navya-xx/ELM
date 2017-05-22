from __future__ import division
from elm_class import ELM
import tensorflow as tf
import numpy as np
# import math
# from tensorflow.examples.tutorials.mnist import input_data

# Basic tf setting
tf.set_random_seed(2016)
sess = tf.Session()

# Get data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# sinc function
x = np.random.uniform(-10,10,10000)
y = np.sin(x) / x
np.where(np.isinf(y), np.zeros_like(y), y)

# divide
rand_training = np.random.choice(10000, 5000, False)
x_training = x[rand_training]
y_training = y[rand_training] + np.random.normal(0, 0.2, 5000)

rand_testing = np.delete(np.arange(0,10000), rand_training)
x_testing = x[rand_testing]
y_testing = y[rand_testing]

# Construct ELM
batch_size = 100
hidden_num = 20
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))
elm = ELM(sess, batch_size, 1, hidden_num, 1)

for i in range(50):
    # one-step feed-forward training
    beta, cost = elm.feed(np.expand_dims(x_training[batch_size*i:batch_size*(i+1)],1), np.expand_dims(y_training[batch_size*i:batch_size*(i+1)],1))
    print(cost)

# testing
elm.test(np.expand_dims(x_training,1), np.expand_dims(y_training,1))