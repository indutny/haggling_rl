import numpy as np

from generator import MAX_TYPES

class RandomAgent:
  def __init__(self):
    self.initial_state = 0
    self.total = 0
    self.target = None

  def step(self, obs, state, mask):
    obs = obs.astype('int32')
    offer = obs[:MAX_TYPES]
    obs = obs[MAX_TYPES:]

    values = obs[:MAX_TYPES]
    obs = obs[MAX_TYPES:]

    counts = obs[:MAX_TYPES]

    if state == self.initial_state:
      self.total = np.sum(counts * values)
      offer_value = np.sum(offer * values)

      # Accept offer
      if offer_value >= self.total / 2.0:
        return 0, None, self.initial_state, None

      # Create target and follow it
      self.target = np.where(values > 0.0, counts, 0)

    # Target reached
    if np.array_equal(offer, self.target):
      return 0, None, self.initial_state, None

    delta = self.target - offer
    for i, d in enumerate(delta):
      if d == 0:
        continue

      if d > 0:
        action = i * 2 + 2
      else:
        action = i * 2 + 1
      break

    return action, None, state + 1, None
