from __future__ import division
import tensorflow as tf
import numpy as np

# CHECK : Constants
omega = 1.

class ELM(object):
  def __init__(self, sess, batch_size, input_len, hidden_num, output_len):
    '''
    Args:
      sess : TensorFlow session.
      batch_size : The batch size (N)
      input_len : The length of input. (L)
      hidden_num : The number of hidden node. (K)
      output_len : The length of output. (O)
    '''
    
    self._sess = sess 
    self._batch_size = batch_size
    self._input_len = input_len
    self._hidden_num = hidden_num
    self._output_len = output_len
    self.psuedoinverse = True

    # for train
    self._x0 = tf.placeholder(tf.float32, [self._batch_size, self._input_len])
    self._t0 = tf.placeholder(tf.float32, [self._batch_size, self._output_len])

    # for test
    self._x1 = tf.placeholder(tf.float32, [None, self._input_len])
    self._t1 = tf.placeholder(tf.float32, [None, self._output_len])

    self._W = tf.Variable(
      tf.random_normal([self._input_len, self._hidden_num]),
      trainable=False, dtype=tf.float32)
    self._b = tf.Variable(
      tf.random_normal([self._hidden_num]),
      trainable=False, dtype=tf.float32)
    self._beta = tf.Variable(
      tf.zeros([self._hidden_num, self._output_len]),
      trainable=False, dtype=tf.float32)
    self._var_list = [self._W, self._b, self._beta]

    self.H0 = tf.matmul(self._x0, self._W) + self._b # N x L
    self.H0_T = tf.transpose(self.H0)

    self.H1 = tf.matmul(self._x1, self._W) + self._b # N x L
    self.H1_T = tf.transpose(self.H1)

    if self.psuedoinverse:
      # beta analytic solution : self._beta_s (K x O)
      if self._input_len < self._hidden_num: # L < K
        identity = tf.constant(np.identity(self._hidden_num), dtype=tf.float32)
        self._beta_s = tf.matmul(tf.matmul(tf.matrix_inverse(
          tf.matmul(self.H0_T, self.H0) + identity/omega),
          self.H0_T), self._t0)
        # _beta_s = (H_T*H + I/om)^(-1)*H_T*T
      else:
        identity = tf.constant(np.identity(self._batch_size), dtype=tf.float32)
        self._beta_s = tf.matmul(tf.matmul(self.H0_T, tf.matrix_inverse(
          tf.matmul(self.H0, self.H0_T)+identity/omega)), self._t0)
        # _beta_s = H_T*(H*H_T + I/om)^(-1)*T

    else:
      # SVD solution for beta
      s, u, v = tf.svd(self.H0, full_matrices=True)
      s_inv = tf.transpose(tf.pad(tf.diag(1/s),[[0, abs(self._batch_size - self._hidden_num)], [0, 0]]))
      self._beta_s = tf.matmul( tf.matmul(tf.matmul(tf.transpose(v), s_inv),tf.transpose(u)), self._t0)

    self._assign_beta = self._beta.assign(self._beta_s)
    self._fx0 = tf.matmul(self.H0, self._beta)
    self._fx1 = tf.matmul(self.H1, self._beta)

    self._cost = tf.losses.mean_squared_error(self._fx0, self._t0)

    self._init = False
    self._feed = False

    # for the mnist test
    # self._correct_prediction = tf.losses.mean_squared_error(self._fx1, self._t1)
    self._accuracy = tf.losses.mean_squared_error(self._fx1, self._t1)

  def feed(self, x, t):
    '''
    Args :
      x : input array (N x L)
      t : label array (N x O)
    '''

    if not self._init : self.init()
    beta, cost = self._sess.run([self._assign_beta, self._cost], {self._x0:x, self._t0:t})
    self._feed = True
    return beta, cost

  def init(self):
    self._sess.run(tf.variables_initializer(self._var_list))
    self._init = True

  def test(self, x, t=None):
    if not self._feed : exit("Not feed-forward trained")
    if t is not None :
      print("Accuracy: {:.9f}".format(self._sess.run(self._accuracy, {self._x1:x, self._t1:t})))
    else :
      return self._sess.run(self._fx1, {self._x1:x})
