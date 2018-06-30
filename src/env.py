import numpy as np
import random

from generator import Generator, MAX_TYPES
from ui import UI

ACTION_SPACE = 5

class Environment:
  def __init__(self,
               types=3, max_rounds=5, min_obj=1, max_obj=6, total=10.0,
               max_steps=50):
    self.opponent_list = []

    self.generator = Generator(types, min_obj, max_obj, total)
    self.ui = UI()

    self.types = types
    self.max_rounds = max_rounds
    self.total = total
    self.max_steps = max_steps

    state = self.reset()

    # +- on each type, left/right, submit button
    self.action_space = ACTION_SPACE
    self.observation_space = state.shape[0]

  def reset(self, force_self=False):
    if len(self.opponent_list) > 0:
      self.player = 'self' if force_self else \
          random.choice([ 'self', 'opponent' ])
      self.opponent = random.choice(self.opponent_list)
      self.opponent_state = self.opponent.initial_state
    else:
      self.player = 'self'
      self.opponent = None
      self.opponent_state = None

    self.steps = 0
    self.done = False
    self.status = 'active'

    objects = self.generator.get()
    self.values = {
      'self': objects['valuations'][0],
      'opponent': objects['valuations'][1],
    }
    self.positions = {
      'self': 0,
      'opponent': 0,
    }
    self.counts = objects['counts']
    self.offer = np.zeros(self.counts.shape, dtype='int32')
    self.proposed_offer = np.copy(self.offer)

    self.ui.initial(self.opponent, self.counts, self.values['self'])

    if self.player is 'opponent':
      self._run_opponent()

      # Opponent accepted zero-offer, can't do nothing!
      if self.done:
        return self.reset()

    if not self.player is 'self':
      raise Exception('Unexpected!')

    return self._make_state()

  def add_opponent(self, opponent):
    self.opponent_list.append(opponent)

  def clear_opponents(self):
    self.opponent_list = []

  def step(self, action):
    player = self.player
    if self.done:
      raise Exception('Already done, can\'t go on')

    done = False

    # Submit
    if action == 0:
      reward, state, done = self._submit()
      if not done and self.player is 'opponent':
        reward, state, done = self._run_opponent()

        # NOTE: `reward` is always for `self`

    # +1/-1
    elif action == 1 or action == 2:
      reward, state = self._make_change(1 if action == 1 else -1)

    # left/right
    elif action == 3 or action == 4:
      reward, state = self._move(-1 if action == 3 else 1)
    else:
      raise Exception('Unknown action {}'.format(action))

    self.done = done
    return state, reward, done, { 'player': player }

  def _make_state(self):
    available_actions = [ 0.0 ] * ACTION_SPACE

    available_actions[0] = 1.0

    # Cell
    pos = self.positions[self.player]
    max_value = self.counts[pos]
    current_value = self.offer[pos]
    if current_value != max_value:
      available_actions[1] = 1.0
    if current_value != 0:
      available_actions[2] = 1.0

    # Movement
    if pos != 0:
      available_actions[3] = 1.0
    if pos != self.types - 1:
      available_actions[4] = 1.0

    res = np.concatenate([
      available_actions,
      [
        float(self.steps) / (2 * self.max_rounds - 1),
        float(pos),
      ],
      self.offer,
      self.values[self.player],
      self.counts,
    ]).astype('float32')
    print(res)
    return res

  def _make_change(self, delta):
    index = self.positions[self.player]

    initial_value = self.offer[index]
    value = initial_value + delta
    max_value = self.counts[index]
    cost = self.values[self.player][index]

    # Clamp
    value = min(value, max_value)
    value = max(value, 0.0)

    self.offer[index] = value

    return -0.007, self._make_state()

  def _move(self, delta):
    initial_pos = self.positions[self.player]
    pos = initial_pos + delta
    pos = max(pos, 0)
    pos = min(pos, MAX_TYPES - 1)
    self.positions[self.player] = pos

    return -0.007, self._make_state()

  def _submit(self):
    # No state change here
    state = self._make_state()
    counter_player = 'opponent' if self.player is 'self' else 'self'

    accepted = np.array_equal(self.offer, self.proposed_offer)

    self.steps += 1

    done = accepted or self.steps == 2 * self.max_rounds
    reward = 0.0
    if accepted:
      self_offer = self.offer
      opponent_offer = self.counts - self_offer
      if not self.player is 'self':
        self_offer, opponent_offer = opponent_offer, self_offer

      self_reward = np.sum(self_offer * self.values['self'], dtype='float32')
      self.ui.accept('self', self_reward)
      opponent_reward = np.sum(opponent_offer * self.values['opponent'],
          dtype='float32')
      self.ui.accept('opponent', opponent_reward)

      # Stimulate bigger relative score
      reward = self_reward - opponent_reward

      # Normalze reward
      reward = reward / self.total
      self.status = 'accepted'
    elif done:
      # Discourage absence of consensus
      reward = -0.25
      self.ui.no_consensus()
      self.status = 'no consensus'
    else:
      self.ui.offer(self.offer, self.counts, self.player)

    # Switch player
    self.player = counter_player
    self.positions[self.player] = 0
    self._inverse_offer()
    self.proposed_offer = np.copy(self.offer)

    # NOTE: reward is actually for `self`, not `opponent`
    return reward, state, done

  def _inverse_offer(self):
    self.offer = self.counts - self.offer

  def _run_opponent(self):
    if not self.player is 'opponent':
      raise Exception('Unexpected!')

    state = self._make_state()

    for i in range(0, self.max_steps):
      action, opponent_state = self.opponent.step(state, self.opponent_state)
      self.opponent_state = opponent_state

      state, reward, done, _ = self.step(action)

      # Submitted
      if not self.player is 'opponent':
        break

    # Didn't finish in time, submit current state
    if self.player is 'opponent':
      _, reward, done, _ = self.step(0)

    # Offer accepted, return reward
    return reward, self._make_state(), done
