import tensorflow as tf
import numpy as np

LSTM_UNITS = 32

class Model:
  def __init__(self, env):
    self.env = env

    self.input = tf.placeholder(tf.float32,
        shape=(None, env.observation_space), name='input')

    self.cell = tf.contrib.rnn.LSTMBlockCell(name='lstm', num_units=LSTM_UNITS)
    state_size = self.cell.state_size

    self.rnn_state = tf.placeholder(tf.float32,
        shape=(None, state_size.c + state_size.h), name='rnn_state')

    state = tf.contrib.rnn.LSTMStateTuple(self.rnn_state[:, :state_size.c],
                                          self.rnn_state[:, state_size.c:])
    x, state = self.cell(self.input, state)

    self.initial_state = np.zeros(self.rnn_state.shape[1])

    # Outputs
    self.action = tf.nn.softmax(
        tf.layers.dense(x, env.action_space, name='action'))
    self.value = tf.squeeze(tf.layers.dense(x, 1, name='value'))
    self.new_state = tf.concat([ state.c, state.h ], axis=-1, name='new_state')

  def fill_feed_dict(self, out, obs, state=None):
    if state is None:
      state = [ self.initial_state ]

    out[self.input] = obs
    out[self.rnn_state] = state
    return out

  def pick_action(self, probs):
    return np.random.choice(self.env.action_space, p=probs)

  def explore(self, sess, game_count=2000, step_count=200):
    state = self.env.reset()
    finished_games = 0
    while finished_games < game_count:
      states, actions, rewards, dones = [], [], [], []

      model_state = self.initial_state
      for i in range(step_count):
        feed_dict = {}
        self.fill_feed_dict(feed_dict, [ state ], [ model_state ])
        action_probs, model_state = sess.run([ self.action, self.new_state ],
            feed_dict=feed_dict)
        action_probs = action_probs[0]
        model_state = model_state[0]

        action = self.pick_action(action_probs)
        next_state, reward, done, _ = self.env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        if done:
          state = self.env.reset()
          finished_games += 1
        else:
          state = next_state

      self.reflect(states, actions, rewards, dones)

  def reflect(self, states, actions, rewards, dones):
    print(np.mean(rewards))
    pass
