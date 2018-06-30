class Agent:
  def __init__(self):
    self.reset_stats()

  def reset_stats(self):
    self.stats = {
      'score': 0,
      'agreements': 0,
      'total': 0,
    }

  def add_score(self, score, agreement):
    self.stats['score'] += score
    self.stats['agreements'] += 1 if agreement else 0
    self.stats['total'] += 1

  def step(self, obs, state):
    raise Exception('Not implemented')
