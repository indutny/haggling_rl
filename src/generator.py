import numpy as np
import random

MAX_TYPES = 3

class Generator:
  def __init__(self, types, min_obj, max_obj, total):
    self.types = types
    self.min_obj = min_obj
    self.max_obj = max_obj
    self.total = total

    self.sets = []

    self._init_sets(np.zeros(MAX_TYPES, dtype='int32'), 0, 0)

  def _init_sets(self, counts, i, total_count):
    remaining = total_count + self.types - i - 1

    min_count = max(1, self.min_obj - remaining)
    max_count = self.max_obj - remaining

    for j in range(min_count, max_count + 1):
      counts[i] = j

      # Recurse
      if i < self.types - 1:
        self._init_sets(counts, i + 1, total_count + j)
        continue

      res = { 'counts': np.copy(counts), 'valuations': [] }
      self._init_valuations(res, np.zeros(MAX_TYPES, dtype='int32'), 0, 0)

      if len(res['valuations']) >= 2:
        self.sets.append(res)

  def _init_valuations(self, obj_set, values, i, total_value):
    count = obj_set['counts'][i]
    max_value = int((self.total - total_value) / count)
    if i == self.types - 1:
      # Not enough value
      if total_value + max_value * count != self.total:
        return

      values[i] = max_value
      obj_set['valuations'].append(np.copy(values))
      return

    for j in range(0, max_value + 1):
      values[i] = j
      self._init_valuations(obj_set, values, i + 1, total_value + j * count)

  def get(self):
    pick = random.sample(self.sets, 1)[0]
    valuations = random.sample(pick['valuations'], 2)

    return {
      'counts': pick['counts'],
      'valuations': valuations,
    }
