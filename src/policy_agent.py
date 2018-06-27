import numpy as np

from generator import MAX_TYPES

class HalfOrEverythingPolicy:
  def __init__(self, values, counts):
    self.values = values
    self.counts = counts
    self.total = np.sum(counts * values)

  def on_offer(self, offer):
    offer_value = np.sum(offer * self.values)

    # Accept offer
    if offer_value >= self.total / 2.0:
      return True, None

    # Generate target
    return False, np.where(self.values > 0.0, self.counts, 0)

class PolicyAgent:
  def __init__(self, policy=HalfOrEverythingPolicy):
    self.name = 'random'
    self.policy = policy

    self.initial_state = (True, None,)
    self.target = None

  def step(self, obs, state):
    obs = obs.astype('int32')
    pos = obs[0]
    obs = obs[1:]

    offer = obs[:MAX_TYPES]
    obs = obs[MAX_TYPES:]

    values = obs[:MAX_TYPES]
    obs = obs[MAX_TYPES:]

    counts = obs[:MAX_TYPES]

    is_first, policy = state
    if policy is None:
      policy = self.policy(values, counts)

    if is_first:
      accept, target = policy.on_offer(offer)

      # Accept offer
      if target:
        return 0, self.initial_state

      self.target = target

    # Target reached
    if np.array_equal(offer, self.target):
      return 0, self.initial_state

    delta = self.target - offer
    for i, d in enumerate(delta):
      if d == 0:
        continue

      if i < pos:
        action = 3
      elif i > pos:
        action = 4
      elif d > 0:
        action = 1
      else:
        action = 2
      break

    return action, (False, policy,)
