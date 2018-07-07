import numpy as np
import random

from generator import Generator, MAX_TYPES
from ui import UI

class Environment:
  def __init__(self,
               types=3, max_rounds=5, min_obj=1, max_obj=6, total=10.0):
    self.opponent_list = []

    self.generator = Generator(types, min_obj, max_obj, total)
    self.ui = UI()

    self.types = types
    self.max_rounds = max_rounds
    self.total = total

    state = self.reset()

    # +- on each type, left/right, submit button
    self.offers = self.generator.offers
    self.action_space = len(self.offers)
    self.observation_space = len(state)

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
    self.last_reward = 0.0

    objects = self.generator.get()
    self.values = {
      'self': objects['valuations'][0],
      'opponent': objects['valuations'][1],
    }
    self.counts = objects['counts']
    self.offer_mask = objects['offer_mask']
    self.proposed_offer = None

    self.ui.initial(self.opponent, self.counts, self.values['self'])

    if self.player is 'opponent':
      self._run_opponent()

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

    if not self.offer_mask[action]:
      raise Exception('Invalid offer {}'.format(action))

    offer = self.offers[action]

    done = False

    reward, state, done = self._submit(offer)
    if not done and self.player is 'opponent':
      # NOTE: `reward` is always for `self`
      reward, state, done = self._run_opponent()

    self.done = done
    return state, reward, done, { 'player': player }

  def bench(self, agent, times=100):
    score = 0.0
    accepted = 0
    for i in range(times):
      is_accepted, delta = self.bench_single(agent)
      score += delta

      if is_accepted:
        accepted += 1

    return {
      'mean': score / float(times),
      'mean_accepted': score / float(accepted),
      'acceptance': float(accepted) / float(times),
    }

  def bench_single(self, agent):
    state = self.reset()
    agent_state = agent.initial_state

    for i in range(self.max_steps):
      action, agent_state = agent.step(state, agent_state)
      state, _, done, _ = self.step(action)
      if done:
        return self.status == 'accepted', self.last_reward

    # Timed out
    return False, 0.0

  def get_offer(self, i):
    return self.offers[i]

  def identify_offer(self, offer):
    for i, other in enumerate(self.offers):
      if np.array_equal(offer, other):
        return i
    raise Exception('Unexpected offer')

  def _make_state(self):
    offer_i = 0
    if not self.proposed_offer is None:
      offer_i = self.identify_offer(self.proposed_offer)

    return np.concatenate([
      self.offer_mask,
      [ offer_i ],
      self.values[self.player],
      self.counts,
    ])

  def _submit(self, offer):
    # No state change here
    state = self._make_state()
    counter_player = 'opponent' if self.player is 'self' else 'self'

    if self.proposed_offer is None:
      accepted = False
    else:
      accepted = np.array_equal(offer, self.proposed_offer)

    self.steps += 1

    done = accepted or self.steps == 2 * self.max_rounds
    reward = 0.0
    if accepted:
      self_offer = offer
      opponent_offer = self.counts - offer
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

      # Just for benching (really messy)
      # TODO(indutny): unmess it
      self.last_reward = reward
    elif done:
      # Discourage absence of consensus
      reward = -1.0
      self.ui.no_consensus()
      self.status = 'no consensus'
    else:
      self.ui.offer(offer, self.counts, self.player)

    # Switch player
    self.player = counter_player
    self.proposed_offer = self.counts - offer

    # NOTE: reward is actually for `self`, not `opponent`
    return reward, state, done

  def _run_opponent(self):
    if not self.player is 'opponent':
      raise Exception('Unexpected!')

    state = self._make_state()
    action, self.opponent_state = self.opponent.step(state, self.opponent_state)

    state, reward, done, _ = self.step(action)

    # Offer accepted, return reward
    return reward, self._make_state(), done
