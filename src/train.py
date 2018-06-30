import os
import time
import random
import tensorflow as tf

from policy_agent import PolicyAgent
from env import Environment
from model import Model

RUN_NAME = os.environ.get('HAGGLE_RUN')
if RUN_NAME is None:
  RUN_NAME = time.strftime('%d.%m.%Y_%H%M%S')
LOG_DIR = os.path.join('.', 'logs', RUN_NAME)
SAVE_DIR = os.path.join('.', 'saves', RUN_NAME)

SAVE_EVERY = 10

MAX_ANTAGONISTS = 1000
NUM_ANTAGONISTS = 8
ANTAGONISTS = []
ANTAGONIST_WEIGHTS = []

# Not really constants, but meh...
EPOCH = 0

env = Environment()

env.add_opponent(PolicyAgent(policy='half_or_all'))
env.add_opponent(PolicyAgent(policy='downsize'))

writer = tf.summary.FileWriter(LOG_DIR)

# Linear schedule
def entropy(game_count):
  if game_count < 100000:
    return 0.01
  elif game_count < 200000:
    return 0.01 - 0.009 * (game_count - 100000.0) / 100000.0
  else:
    return 0.001

with tf.Session() as sess:
  model = Model(env, sess, writer, name='haggle')
  saver = tf.train.Saver(max_to_keep=10000, name=RUN_NAME)

  for i in range(NUM_ANTAGONISTS):
    antagonist = Model(env, sess, None, name='antagonist')
    env.add_opponent(antagonist)
    ANTAGONISTS.append(antagonist)

  sess.run(tf.global_variables_initializer())

  # Run preliminary games to get antogonist started
  while True:
    game_off = 0
    model.explore(game_count=1000, game_off=game_off, entropy_scale=entropy)
    game_off += 1000
    EPOCH += 1

    if EPOCH % SAVE_EVERY == 0:
      saver.save(sess, os.path.join(SAVE_DIR, '{:08d}'.format(EPOCH)))

    weights = model.save_weights(sess)
    ANTAGONIST_WEIGHTS.append({
      'epoch': EPOCH,
      'weights': weights
    })
    if len(ANTAGONIST_WEIGHTS) > MAX_ANTAGONISTS:
      ANTAGONIST_WEIGHTS.pop(random.randrange(len(ANTAGONIST_WEIGHTS)))

    if len(ANTAGONIST_WEIGHTS) > 0:
      for antagonist in ANTAGONISTS:
        save = random.choice(ANTAGONIST_WEIGHTS)
        print('Loading epoch {} into antagonist'.format(save['epoch']))
        antagonist.load_weights(sess, save['weights'])
        antagonist.set_version(save['epoch'])
