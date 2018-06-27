import tensorflow as tf
import numpy as np

LSTM_UNITS = 32
ENTROPY_SCALE = 0.01
VALUE_SCALE = 0.5
MAX_STEPS = 100
LR = 0.001
GRAD_CLIP = 0.5

class Model:
  def __init__(self, env, sess, writer, name='haggle'):
    self.original_name = name
    self.version = 0

    self.name = name

    self.scope = tf.VariableScope(reuse=False, name=name)
    self.env = env
    self.sess = sess
    self.writer = writer
    self.writer_step = 0

    with tf.variable_scope(self.scope):
      self.input = tf.placeholder(tf.float32,
          shape=(None, env.observation_space), name='input')

      self.cell = tf.contrib.rnn.LSTMBlockCell(name='lstm', \
          num_units=LSTM_UNITS)
      state_size = self.cell.state_size

      self.rnn_state = tf.placeholder(tf.float32,
          shape=(None, state_size.c + state_size.h), name='rnn_state')

      state = tf.contrib.rnn.LSTMStateTuple(c=self.rnn_state[:, :state_size.c],
                                            h=self.rnn_state[:, state_size.c:])
      x, state = self.cell(self.input, state)

      self.initial_state = np.zeros(self.rnn_state.shape[1])

      # Outputs
      raw_action = tf.layers.dense(x, env.action_space, name='action')
      self.action = tf.nn.softmax(raw_action)
      self.value = tf.squeeze(tf.layers.dense(x, 1, name='value'))
      self.mean_value = tf.reduce_mean(self.value, name='mean_value')
      self.new_state = tf.concat([ state.c, state.h ], axis=-1, \
          name='new_state')

      # Losses
      self.true_value = tf.placeholder(tf.float32,
          shape=(None,), name='true_value')
      self.selected_action = tf.placeholder(tf.int32,
          shape=(None,), name='selected_action')

      self.entropy = -tf.reduce_mean(
          tf.reduce_sum(self.action * tf.log(self.action), axis=-1),
          name='entropy')

      advantage = self.true_value - self.value
      self.value_loss = tf.reduce_mean(advantage ** 2, name='value_loss') / 2.0

      action_gain = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=self.selected_action,
          logits=raw_action,
          name='action_gain')
      self.policy_loss = tf.reduce_mean(action_gain * self.true_value,
          name='policy_loss')

      optimizer = tf.train.AdamOptimizer(LR)

      self.loss = self.policy_loss + self.value_loss * VALUE_SCALE + \
          self.entropy * ENTROPY_SCALE

      variables = tf.trainable_variables()
      grads = tf.gradients(self.loss, variables)
      grads, grad_norm = tf.clip_by_global_norm(grads, GRAD_CLIP)
      grads = list(zip(grads, variables))
      self.grad_norm = grad_norm
      self.train = optimizer.apply_gradients(grads_and_vars=grads)

  def copy_from(self, other):
    self_vars = self.scope.trainable_variables()
    other_vars = other.scope.trainable_variables()

    ops = []
    for self_var, other_var in zip(self_vars, other_vars):
      ops.append(self_var.assign(other_var))
    return ops

  def set_version(self, version):
    self.version = version
    self.name = '{}_v{}'.format(self.original_name, self.version)

  def fill_feed_dict(self, out, obs, state=None):
    if state is None:
      state = [ self.initial_state ]

    out[self.input] = obs
    out[self.rnn_state] = state
    return out

  def step(self, obs, state):
    feed_dict = {}
    self.fill_feed_dict(feed_dict, [ obs ], [ state ])
    action_probs, next_state = self.sess.run([ self.action, self.new_state ],
        feed_dict=feed_dict)
    action = self.pick_action(action_probs[0])
    return action, next_state[0]

  def pick_action(self, probs):
    return np.random.choice(self.env.action_space, p=probs)

  def explore(self, game_count=20000, step_count=256):
    state = self.env.reset()
    finished_games = 0
    while finished_games < game_count:
      print('collecting...')
      states, model_states, actions, rewards, dones = [], [], [], [], []

      model_state = self.initial_state
      steps = 0
      for i in range(step_count):
        action, next_model_state = self.step(state, model_state)

        next_state, reward, done, _ = self.env.step(action)

        if not done and steps > MAX_STEPS:
          print('timed out...')
          reward = -2.0
          done = True

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        model_states.append(model_state)

        if done:
          state = self.env.reset()
          steps = 0
          model_state = self.initial_state
          finished_games += 1
        else:
          state = next_state
          steps += 1
          model_state = next_model_state

      print('reflecting...')
      self.reflect(states, model_states, actions, rewards, dones)

  def estimate_rewards(self, rewards, dones, gamma=0.99):
    estimates = np.zeros(len(rewards), dtype='float32')
    future = 0.0

    for i, reward in reversed(list(enumerate(rewards))):
      if dones[i]:
        future = 0.0
      future *= gamma
      estimates[i] = reward + future
      future += reward

    return estimates

  def reflect(self, states, model_states, actions, rewards, dones):
    estimates = self.estimate_rewards(rewards, dones)

    feed_dict = {
      self.input: states,
      self.rnn_state: model_states,
      self.selected_action: actions,
      self.true_value: estimates,
    }

    tensors = [
        self.train, self.loss, self.entropy, self.value_loss, self.policy_loss,
        self.mean_value, self.grad_norm,
    ]
    _, loss, entropy, value_loss, policy_loss, value, grad_norm = self.sess.run(
        tensors, feed_dict=feed_dict)

    metrics = {
      'grad_norm': grad_norm,
      'loss': loss,
      'entropy': entropy,
      'value_loss': value_loss,
      'policy_loss': policy_loss,
      'true_value': np.mean(estimates),
      'value': value,
    }
    self.log_summary(metrics)

  def log_summary(self, metrics):
    summary = tf.Summary()
    for key in metrics:
      value = metrics[key]
      summary.value.add(tag='train/{}'.format(key), simple_value=value)
    self.writer.add_summary(summary, self.writer_step)
    self.writer.flush()
    self.writer_step += 1
