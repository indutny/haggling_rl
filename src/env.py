import numpy as np
import random

from generator import Generator, MAX_TYPES
from ui import UI

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
    self.action_space = 5
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
        reward = -reward

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
    return np.concatenate([
      [ self.positions[self.player] ],
      self.offer,
      self.values[self.player],
      self.counts,
    ]).astype('float32')

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

    reward = 0.0
    return reward, self._make_state()

  def _move(self, delta):
    initial_pos = self.positions[self.player]
    pos = initial_pos + delta
    pos = max(pos, 0)
    pos = min(pos, MAX_TYPES - 1)
    self.positions[self.player] = pos

    reward = 0.0
    return reward, self._make_state()

  def _submit(self):
    # No state change here
    state = self._make_state()
    counter_player = 'opponent' if self.player is 'self' else 'self'

    accepted = np.array_equal(self.offer, self.proposed_offer)

    self.steps += 1

    done = accepted or self.steps == 2 * self.max_rounds
    reward = 0.0
    if accepted:
      reward = np.sum(self.offer * self.values[self.player], dtype='float32')
      self.ui.accept(self.player, reward)
      counter_reward = np.sum(
          (self.counts - self.offer) * self.values[counter_player],
          dtype='float32')
      self.ui.accept(counter_player, counter_reward)

      # Stimulate bigger absolute score
      if self.player is 'self':
        reward *= 1.25
      else:
        reward *= 1.25

      # ...and bigger relative score
      reward -= counter_reward
      reward = reward / self.total
    elif done:
      # Slightly discourage absence of consensus
      reward = -0.15 if self.player is 'self' else 0.15
      self.ui.no_consensus()
    else:
      self.ui.offer(self.offer, self.counts, self.player)

    # Switch player
    self.player = counter_player
    self.positions[self.player] = 0
    self._inverse_offer()
    self.proposed_offer = np.copy(self.offer)

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
