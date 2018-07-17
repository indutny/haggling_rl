import numpy as np
import random

from generator import MAX_TYPES
from agent import Agent

class BasePolicy:
  def __init__(self, values, counts):
    self.values = values
    self.counts = counts
    self.max_types = len(self.counts)
    self.total = np.sum(counts * values)

class HalfOrAllPolicy(BasePolicy):
  def on_offer(self, offer):
    offer_value = np.sum(offer * self.values)

    # Accept offer
    if offer_value >= self.total / 2.0:
      return True, None

    # Generate target
    return False, np.where(self.values > 0.0, self.counts, 0)

class DownsizePolicy(BasePolicy):
  def __init__(self, *args, **kwargs):
    super(DownsizePolicy, self).__init__(*args, **kwargs)

    self.round = 0

  def on_offer(self, offer):
    self.round += 1

    alpha = 1 - min(5, self.round) / 5.0
    half = self.total / 2.0
    minimum = (self.total - half) * alpha + half

    offer_value = np.sum(offer * self.values)

    # Accept offer
    if offer_value >= minimum:
      return True, None

    offers = []
    self.find_offers(offers, np.zeros(self.counts.shape), minimum, 0, 0.0)

    min_value = float('inf')
    for offer_value, offer in offers:
      if offer_value > min_value:
        continue
      min_value = offer_value

    min_offers = [ offer for value, offer in offers if value == min_value ]
    offer = random.choice(min_offers)

    # Generate target
    return False, offer

  def find_offers(self, offers, offer, minimum, i, total):
    if i == len(self.counts):
      return

    if self.values[i] == 0.0:
      return self.find_offers(offers, offer, minimum, i + 1, total)

    for j in range(0, self.counts[i] + 1):
      offer[i] = j
      offer_value = total + j * self.values[i]
      if offer_value >= minimum:
        offers.append((offer_value, np.copy(offer),))

      self.find_offers(offers, offer, minimum, i + 1, offer_value)

class AltruistPolicy(BasePolicy):
  def on_offer(self, offer):
    offer_value = np.sum(offer * self.values)
    if offer_value > 0:
      return True, None

    min_value_i = np.argmin(
        np.where(self.values > 0, self.values, float('inf')))
    counter_offer = np.zeros(self.counts.shape)
    counter_offer[min_value_i] = 1

    return False, counter_offer

class GreedyPolicy(BasePolicy):
  def on_offer(self, offer):
    offer_value = np.sum(offer * self.values)
    if offer_value == self.total:
      return True, None

    return False, np.copy(self.counts)

class StubbornPolicy(BasePolicy):
  def on_offer(self, offer):
    counter_offer = np.zeros(self.counts.shape)
    for i, max_count in enumerate(self.counts):
      counter_offer[i] = random.randint(0, max_count)
    return False, counter_offer

class EstimatorPolicy(BasePolicy):
  def __init__(self, *args, **kwargs):
    super(EstimatorPolicy, self).__init__(*args, **kwargs)

    self.possible_values = []
    self.possible_offers = []

    self.fill_values(np.zeros(self.max_types, dtype='int32'), 0, 0)
    self.fill_offers(np.zeros(self.max_types, dtype='int32'), 0)

    # Opponent can't have same values as ourselves
    self.possible_values = [
      v
      for v in self.possible_values if not np.array_equal(v, self.values)
    ]

    self.possible_offers = [
      o
      for o in self.possible_offers if self.offer_value(o, self.values) >= 0.5
    ]

    self.cross_map = []
    for o in self.possible_offers:
      values = []
      for v in self.possible_values:
        values.append({
          'self': self.offer_value(o, self.values),
          'opponent': self.offer_value(self.invert_offer(o), v),
        })
      entry = { 'offer': o, 'values': values }
      self.cross_map.append(entry)

    self.past_offers = []
    self.used = [ False for _ in self.possible_offers ]

  def invert_offer(self, offer):
    return self.counts - offer

  def offer_value(self, offer, values):
    return np.sum(offer * values, dtype='float32') / self.total

  def fill_values(self, values, i, total):
    count = self.counts[i]
    max_value = (self.total - total) // count
    if i == self.max_types - 1:
      if total + max_value * count == self.total:
        values[i] = max_value
        self.possible_values.append(np.copy(values))
      return

    for j in range(max_value):
      values[i] = j
      self.fill_values(values, i + 1, total + j * count)

  def fill_offers(self, offer, i):
    if i == self.max_types:
      self.possible_offers.append(np.copy(offer))
      return

    for j in range(self.counts[i] + 1):
      offer[i] = j
      self.fill_offers(offer, i + 1)

  def on_offer(self, offer):
    self.past_offers.append(self.invert_offer(offer))
    estimates = self.estimate(self.past_offers)

    def score_each(entry):
      res = 0.0
      for i, value in enumerate(entry['values']):
        estimate = estimates[i]

        # Unlikely to be accepted
        if value['opponent'] < 0.5:
          res += 0.1
          continue

        delta = value['self'] - value['opponent']

        res += estimate * delta

      return res

    scores = map(score_each, self.cross_map)

    max_score = float('-inf')
    max_i = None

    for i, score in enumerate(scores):
      if not self.used[i] and score > max_score:
        max_score = score
        max_i = i

    # Can't find right offer
    if max_i is None:
      return False, self.counts

    result = self.possible_offers[max_i]
    value = self.offer_value(result, self.values)
    proposed_value = self.offer_value(offer, self.values)
    if value == proposed_value:
      return True, None

    self.used[max_i] = True
    return False, result

  def estimate(self, past_offers):
    scores = []
    for values in self.possible_values:
      score = 0.0
      for o in past_offers:
        score += max(0.0, self.offer_value(o, values) - 0.5)
      scores.append(score)
    return scores

class PolicyAgent(Agent):
  def __init__(self, env, policy):
    super(PolicyAgent, self).__init__()

    self.env = env

    if policy is 'downsize':
      self.policy = DownsizePolicy
    elif policy is 'half_or_all':
      self.policy = HalfOrAllPolicy
    elif policy is 'altruist':
      self.policy = AltruistPolicy
    elif policy is 'greedy':
      self.policy = GreedyPolicy
    elif policy is 'stubborn':
      self.policy = StubbornPolicy
    elif policy is 'estimator':
      self.policy = EstimatorPolicy
    else:
      self.policy = policy

    self.name = 'agent_' + self.policy.__name__

    self.target = None

  def build_initial_state(self, context):
    self.values = context[:MAX_TYPES]
    self.counts = context[MAX_TYPES:]

  def step(self, obs, policy):
    available_offers = obs[:self.env.action_space]
    obs = obs[len(available_offers):]

    proposed_offer = self.env.get_offer(int(obs[0]))

    # Initial offer
    if proposed_offer is True:
      proposed_offer = np.zeros(MAX_TYPES, dtype='int32')

    obs = obs[1:]

    if policy is None:
      policy = self.policy(self.values, self.counts)

    accept, target = policy.on_offer(proposed_offer)

    # Accept offer
    if accept:
      return True, policy

    return target, policy
