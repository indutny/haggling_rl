import tensorflow as tf
import numpy as np

from agent import Agent

PRE_WIDTH = None
LSTM_UNITS = 64
VALUE_SCALE = 0.5
MAX_STEPS = 100
LR = 0.001
GRAD_CLIP = 0.5
PPO_EPSILON = 0.1

def default_entropy_schedule(game_count):
  return 0.01

class Model(Agent):
  def __init__(self, policy):
    super(Model, self).__init__()

  def __init__(self, env, sess, writer, name='haggle'):
    self.original_name = name
    self.version = 0

    self.name = name

    self.observation_space = env.observation_space
    self.action_space = env.action_space

    self.scope = tf.VariableScope(reuse=False, name=name)
    self.sess = sess
    self.writer = writer
    self.writer_step = 0

    with tf.variable_scope(self.scope):
      self.input = tf.placeholder(tf.float32,
          shape=(None, self.observation_space), name='input')

      self.cell = tf.contrib.rnn.LSTMBlockCell(name='lstm', \
          num_units=LSTM_UNITS)
      state_size = self.cell.state_size

      self.rnn_state = tf.placeholder(tf.float32,
          shape=(None, state_size.c + state_size.h), name='rnn_state')

      available_actions, x = tf.split(self.input, [
        self.action_space, self.observation_space - self.action_space ], axis=1)

      if not PRE_WIDTH is None:
        x = tf.layers.dense(x, PRE_WIDTH, name='preprocess',
                            activation=tf.nn.relu)

      state = tf.contrib.rnn.LSTMStateTuple(c=self.rnn_state[:, :state_size.c],
                                            h=self.rnn_state[:, state_size.c:])
      x, state = self.cell(x, state)

      self.initial_state = np.zeros(self.rnn_state.shape[1], dtype='float32')

      # Outputs
      raw_action = tf.layers.dense(x, self.action_space, name='action')
      raw_action *= available_actions
      raw_action += (1.0 - available_actions) * -1e23

      self.action_probs = tf.nn.softmax(raw_action, name='action_probs')
      action_dist = tf.distributions.Categorical(probs=self.action_probs)

      self.action = action_dist.sample()

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
          tf.reduce_sum(self.action_probs * tf.log(self.action_probs + 1e-20),
              axis=-1),
          name='entropy')

      online_advantage = self.true_value - self.value
      self.value_loss = tf.reduce_mean(online_advantage ** 2,
          name='value_loss') / 2.0

      action_one_hot = tf.one_hot(self.selected_action,
          depth=self.action_probs.shape[-1],
          dtype='float32',
          name='action_one_hot')
      current_prob = tf.reduce_sum(
          action_one_hot * self.action_probs,
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

  def step(self, obs, state):
    feed_dict = {}
    self.fill_feed_dict(feed_dict, [ obs ], [ state ])
    tensors = [ self.action, self.new_state ]

    action, next_state = self.sess.run(tensors, feed_dict=feed_dict)
    return action, next_state[0]

  def multi_step(self, env_states, model_states):
    feed_dict = {
      self.input: env_states,
      self.rnn_state: model_states,
    }
    tensors = [ self.action, self.action_probs, self.new_state, self.value ]
    out = self.sess.run(tensors, feed_dict=feed_dict)

    actions, action_probs, next_model_states, values = out
    action_probs = [
        probs[action] for action, probs in zip(actions, action_probs) ]

    return actions, next_model_states, values, action_probs

  def game(self, env_list):
    log = [ {
      'env_states': [],
      'actions': [],
      'action_probs': [],
      'values': [],
      'rewards': [],
      'dones': [],
      'model_states': [],
      'statuses': [],
    } for i in range(len(env_list)) ]

    model_states = [ self.initial_state for _ in env_list ]
    env_states = [ env.reset() for env in env_list ]

    steps = 0
    while True:
      actions, next_model_states, values, action_probs = \
          self.multi_step(env_states, model_states)

      next_env_states = []
      rewards = []
      dones = []
      statuses = []
      for env, env_state, action in zip(env_list, env_states, actions):
        if not env.done:
          next_env_state, reward, done, _ = env.step(action)
        else:
          next_env_state, reward, done = env_state, 0.0, True

        next_env_states.append(next_env_state)
        rewards.append(reward)
        dones.append(done)
        statuses.append(env.status)

      steps += 1
      if steps > MAX_STEPS:
        width = len(env_list)

        rewards = [ -1.5 ] * width
        dones = [ True ] * width
        statuses = [ 'timeout' ] * width

      zipped = zip(env_states, rewards, dones, statuses, model_states, actions,
          action_probs, values)
      for i, t in enumerate(zipped):
        env_state, reward, done, status, model_state, action, action_prob, \
            value = t

        entry = log[i]
        entry['env_states'].append(env_state)
        entry['rewards'].append(reward)
        entry['dones'].append(done)
        entry['statuses'].append(status)
        entry['model_states'].append(model_state)
        entry['actions'].append(action)
        entry['action_probs'].append(action_prob)
        entry['values'].append(value)

      # Update states
      env_states = next_env_states
      model_states = next_model_states

      has_pending = False in dones
      if not has_pending:
        # All completed
        break

      if steps > MAX_STEPS:
        # All timed out
        break

    return log

  def explore(self, env_list, game_count=1024, reflect_every=128, game_off=0, \
              entropy_schedule=default_entropy_schedule):
    finished_games = 0
    while finished_games < game_count:
      reflect_target = min(game_count - finished_games, reflect_every)

      games = self.collect(env_list, reflect_target)
      finished_games += reflect_target

      self.reflect(games,
          entropy_coeff=entropy_schedule(game_off + finished_games))

  def collect(self, env_list, count):
    res = {
      'env_states': [],
      'model_states': [],
      'actions': [],
      'action_probs': [],
      'values': [],
      'rewards': [],
      'dones': [],
      'acceptance': [],
      'steps_per_game': [],
    }

    if count % len(env_list) != 0:
      raise Exception('Number of games is not divisible by concurrency')

    for game_index in range(count // len(env_list)):
      log = self.game(env_list)

      global_max_steps = len(log[0]['dones'])
      for entry in log:
        dones = entry['dones']
        max_steps = None
        for i in range(global_max_steps):
          if dones[i]:
            max_steps = i + 1
            break

        res['env_states'] += entry['env_states'][:max_steps]
        res['model_states'] += entry['model_states'][:max_steps]
        res['actions'] += entry['actions'][:max_steps]
        res['action_probs'] += entry['action_probs'][:max_steps]
        res['values'] += entry['values'][:max_steps]
        res['rewards'] += entry['rewards'][:max_steps]
        res['dones'] += entry['dones'][:max_steps]

        statuses = entry['statuses'][:max_steps]

        res['acceptance'].append(statuses[max_steps - 1] is 'accepted')
        res['steps_per_game'].append(max_steps)

    res['steps_per_game'] = np.mean(res['steps_per_game'])
    res['acceptance'] = np.mean(res['acceptance'])

    return res

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
      self.input: games['env_states'],
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
