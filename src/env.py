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

  def step(self, offer):
    player = self.player
    if self.done:
      raise Exception('Already done, can\'t go on')

    if isinstance(offer, int):
      offer = self.offers[offer]

    for val, max in zip(offer, self.counts):
      if val < 0 or val > max:
        raise Exception('Invalid offer')

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

    while True:
      action, agent_state = agent.step(state, agent_state)
      state, _, done, _ = self.step(action)
      if done:
        return self.status == 'accepted', self.last_reward

    # Timed out
    return False, 0.0

  def _make_state(self):
    proposed_offer = self.proposed_offer
    if proposed_offer is None:
      proposed_offer = np.zeros(MAX_TYPES, dtype='int32')

    return np.concatenate([
      self.offer_mask,
      proposed_offer,
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

      # Normalze rewards
      self_reward_p = self_reward / self.total
      opponent_reward_p = opponent_reward / self.total

      if self_reward_p < 0.7:
        if self_reward_p < 0:
          raise Exception('Unexpected reward')

        opponent_reward_p *= (1.7 - self.reward_p)

      # Stimulate bigger relative score
      reward = self_reward_p - opponent_reward_p

      self.status = 'accepted'

      # Just for benching (really messy)
      # TODO(indutny): unmess it
      self.last_reward = self_reward
    elif done:
      # Discourage absence of consensus
      reward = -1.0
      self.last_reward = 0.0
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
