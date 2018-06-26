import numpy as np

from generator import MAX_TYPES

class RandomAgent:
  def __init__(self):
    self.initial_state = 0
    self.total = 0
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

    if state == self.initial_state:
      self.total = np.sum(counts * values)
      offer_value = np.sum(offer * values)

      # Accept offer
      if offer_value >= self.total / 2.0:
        return 0, self.initial_state 

      # Create target and follow it
      self.target = np.where(values > 0.0, counts, 0)

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

    return action, state + 1
