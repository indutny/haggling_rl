import numpy as np
import random

from generator import MAX_TYPES
from agent import Agent

class BasePolicy:
  def __init__(self, values, counts):
    self.values = values
    self.counts = counts
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
    else:
      self.policy = policy

    self.name = 'agent_' + self.policy.__name__

    self.initial_state = None
    self.target = None

  def step(self, obs, policy):
    available_offers = obs[:self.env.action_space]
    obs = obs[len(available_offers):]

    proposed_offer = obs[:MAX_TYPES]
    obs = obs[len(proposed_offer):]

    values = obs[:MAX_TYPES]
    obs = obs[len(values):]

    counts = obs[:MAX_TYPES]
    obs = obs[len(counts):]

    if policy is None:
      policy = self.policy(values, counts)

    accept, target = policy.on_offer(proposed_offer)

    # Accept offer
    if accept:
      return proposed_offer, self.initial_state

    return target, policy
