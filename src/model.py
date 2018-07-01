import tensorflow as tf
import numpy as np

from agent import Agent

PRE_WIDTH = 64
LSTM_UNITS = 128
VALUE_SCALE = 0.5
MAX_STEPS = 100
LR = 0.001
GRAD_CLIP = 0.5
PPO_EPSILON = 0.1

def default_entropy_schedule(game_count):
  return 0.005

class Model(Agent):
  def __init__(self, policy):
    super(Model, self).__init__()

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

      available_actions, x = tf.split(self.input, [
        env.action_space, env.observation_space - env.action_space ], axis=1)

      x = tf.layers.dense(x, PRE_WIDTH, name='preprocess',
                          activation=tf.nn.relu)

      state = tf.contrib.rnn.LSTMStateTuple(c=self.rnn_state[:, :state_size.c],
                                            h=self.rnn_state[:, state_size.c:])
      x, state = self.cell(x, state)

      self.initial_state = np.zeros(self.rnn_state.shape[1], dtype='float32')
      self.initial_state_var = tf.Variable(self.initial_state,
          name='initial_state')

      self.new_initial_state = tf.placeholder(tf.float32,
          shape=self.initial_state_var.shape, name='new_initial_state')
      self.update_initial_state = self.initial_state_var.assign(
          self.new_initial_state)

      # Outputs
      raw_action = tf.layers.dense(x, env.action_space, name='action')
      raw_action *= available_actions
      raw_action += (1.0 - available_actions) * -1e23

      self.action = tf.nn.softmax(raw_action, name='action_probs')
      self.value = tf.squeeze(tf.layers.dense(x, 1, name='value'))
      self.mean_value = tf.reduce_mean(self.value, name='mean_value')
      self.new_state = tf.concat([ state.c, state.h ], axis=-1, \
          name='new_state')

      # Losses
      self.true_value = tf.placeholder(tf.float32,
          shape=(None,), name='true_value')
      self.past_value = tf.placeholder(tf.float32,
          shape=(None,), name='past_value')
      self.past_prob = tf.placeholder(tf.float32,
          shape=(None,), name='past_prob')
      self.selected_action = tf.placeholder(tf.int32,
          shape=(None,), name='selected_action')
      self.entropy_coeff = tf.placeholder(tf.float32, shape=(),
          name='entropy_coeff')

      self.entropy = -tf.reduce_mean(
          tf.reduce_sum(self.action * tf.log(self.action + 1e-20), axis=-1),
          name='entropy')

      online_advantage = self.true_value - self.value
      self.value_loss = tf.reduce_mean(online_advantage ** 2,
          name='value_loss') / 2.0

      action_one_hot = tf.one_hot(self.selected_action,
          depth=self.action.shape[-1],
          dtype='float32',
          name='action_one_hot')
      current_prob = tf.reduce_sum(
          action_one_hot * self.action,
          axis=-1,
          name='current_prob')

      prob_ratio = current_prob / self.past_prob
      clipped_ratio = tf.clip_by_value(prob_ratio, 1.0 - PPO_EPSILON,
          1.0 + PPO_EPSILON, name='clipped_ratio')

      offline_advantage = self.true_value - self.past_value
      policy_loss = tf.minimum(
          prob_ratio * offline_advantage,
          clipped_ratio * offline_advantage)
      policy_loss = -tf.reduce_mean(policy_loss,
          name='policy_loss')
      self.policy_loss = policy_loss

      optimizer = tf.train.AdamOptimizer(LR)

      self.loss = self.policy_loss + self.value_loss * VALUE_SCALE - \
          self.entropy * self.entropy_coeff

      variables = tf.trainable_variables()
      grads = tf.gradients(self.loss, variables)
      grads, grad_norm = tf.clip_by_global_norm(grads, GRAD_CLIP)
      grads = list(zip(grads, variables))
      self.grad_norm = grad_norm
      self.train = optimizer.apply_gradients(grads_and_vars=grads)

      # Weight loading
      self.trainable_variables = self.scope.trainable_variables()

      self.weight_placeholders = {}
      self.load_ops = []
      for var in self.trainable_variables:
        name = var.name.split(':', 1)[0]
        name = name.split('/', 1)[1]
        placeholder = tf.placeholder(var.dtype,
            shape=var.shape,
            name='{}/placeholder'.format(name))
        self.weight_placeholders[name] = placeholder
        self.load_ops.append(var.assign(placeholder))

  def save_weights(self, sess):
    values = sess.run(self.trainable_variables)
    out = {}
    for var, value in zip(self.trainable_variables, values):
      name = var.name.split(':', 1)[0]
      name = name.split('/', 1)[1]
      out[name] = value
    return out

  def load_weights(self, weights):
    feed_dict = {}
    for name, value in weights.items():
      if name in self.weight_placeholders:
        feed_dict[self.weight_placeholders[name]] = value
    return feed_dict, self.load_ops

  def set_version(self, version):
    self.version = version
    self.name = '{}_v{}'.format(self.original_name, self.version)

  def fill_feed_dict(self, out, obs, state=None):
    if state is None:
      state = [ self.initial_state ]

    out[self.input] = obs
    out[self.rnn_state] = state
    return out

  def step(self, obs, state, a2c=False):
    feed_dict = {}
    self.fill_feed_dict(feed_dict, [ obs ], [ state ])
    tensors = [ self.action, self.new_state ]
    if a2c:
      tensors.append(self.value)

    out = self.sess.run(tensors, feed_dict=feed_dict)
    if not a2c:
      out.append(None)

    action_probs, next_state, value = out
    action = self.pick_action(action_probs[0])

    if a2c:
      return action, next_state[0], value, action_probs[0][action]
    else:
      return action, next_state[0]

  def pick_action(self, probs):
    return np.random.choice(self.env.action_space, p=probs)

  def explore(self, game_count=1024, reflect_every=128, game_off=0, \
              entropy_schedule=default_entropy_schedule):
    finished_games = 0
    while finished_games < game_count:
      reflect_target = min(game_count - finished_games, reflect_every)

      games = self.collect(reflect_target)
      finished_games += reflect_target

      self.reflect(games,
          entropy_coeff=entropy_schedule(game_off + finished_games))

  def collect(self, count):
    states, model_states, actions, probs, values, rewards, dones, accepted = \
        [], [], [], [], [], [], [], []

    model_state = self.initial_state
    state = self.env.reset()
    finished_games = 0

    steps = 0
    steps_per_game = []
    while finished_games < count:
      action, next_model_state, value, action_prob = \
          self.step(state, model_state, a2c=True)

      next_state, reward, done, _ = self.env.step(action)
      status = self.env.status

      if not done and steps > MAX_STEPS:
        reward = -1.0
        status = 'timeout'
        done = True

      states.append(state)
      actions.append(action)
      probs.append(action_prob)
      values.append(value)
      rewards.append(reward)
      dones.append(done)
      model_states.append(model_state)

      model_state = next_model_state

      if done:
        state = self.env.reset()
        steps_per_game.append(steps)
        steps = 0
        finished_games += 1

        accepted.append(status is 'accepted')
      else:
        state = next_state
        steps += 1

    steps_per_game = np.mean(steps_per_game)

    # Save new initial state
    self.initial_state = model_state
    self.sess.run(self.update_initial_state, feed_dict={
      self.new_initial_state: self.initial_state,
    })

    return {
      'states': states,
      'model_states': model_states,
      'actions': actions,
      'action_probs': probs,
      'values': values,
      'rewards': rewards,
      'dones': dones,
      'acceptance': np.mean(accepted),
      'steps_per_game': steps_per_game
    }

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

  def reflect(self, games, entropy_coeff):
    estimates = self.estimate_rewards(games['rewards'], games['dones'])

    feed_dict = {
      self.input: games['states'],
      self.rnn_state: games['model_states'],
      self.selected_action: games['actions'],
      self.true_value: estimates,
      self.past_value: games['values'],
      self.past_prob: games['action_probs'],
      self.entropy_coeff: entropy_coeff,
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
      'entropy_coeff': entropy_coeff,
      'value_loss': value_loss,
      'policy_loss': policy_loss,
      'steps_per_game': games['steps_per_game'],
      'acceptance': games['acceptance'],
      'reward': np.mean(estimates),
      'max_reward': np.max(estimates),
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
