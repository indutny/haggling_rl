import tensorflow as tf
import numpy as np

LSTM_UNITS = 32

class Model:
  def __init__(self, env):
    self.input = tf.placeholder(tf.float32,
        shape=(None, env.observation_space), name='input')

    self.cell = tf.contrib.rnn.LSTMBlockCell(name='lstm', num_units=LSTM_UNITS)
    state_size = self.cell.state_size

    self.rnn_state = tf.placeholder(tf.float32,
        shape=(None, state_size.c + state_size.h), name='rnn_state')

    state = tf.contrib.rnn.LSTMStateTuple(self.rnn_state[:, :state_size.c],
                                          self.rnn_state[:, state_size.c:])
    x, state = self.cell(self.input, state)

    # Outputs
    self.action = tf.nn.softmax(
        tf.layers.dense(x, env.action_space, name='action'))
    self.value = tf.squeeze(tf.layers.dense(x, 1, name='value'))
    self.new_state = tf.concat([ state.c, state.h ], axis=-1, name='new_state')

  def fill_feed_dict(self, out, obs, state=None):
    if state is None:
      state = np.zeros([1, self.rnn_state.shape[1] ])

    out[self.input] = obs
    out[self.rnn_state] = state
    return out
