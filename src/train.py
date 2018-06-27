import os
import time
import tensorflow as tf

from random_agent import RandomAgent
from env import Environment
from model import Model

RUN_NAME = os.environ.get('HAGGLE_RUN')
if RUN_NAME is None:
  RUN_NAME = time.asctime()
LOG_DIR = os.path.join('.', 'logs', RUN_NAME)

NUM_ANTAGONISTS = 8
ANTAGONIST_INDEX = 0
EPOCH = 0

env = Environment()

env.add_opponent(RandomAgent())

writer = tf.summary.FileWriter(LOG_DIR)

with tf.Session() as sess:
  model = Model(env, sess, writer, name='haggle')

  antagonists = []
  antagonists_copy_ops = []
  for i in range(NUM_ANTAGONISTS):
    antagonist = Model(env, sess, writer, name='haggle_antagonist_{}'.format(i))
    copy_ops = antagonist.copy_from(model)

    antagonists.append(antagonist)
    antagonists_copy_ops.append(copy_ops)

  sess.run(tf.global_variables_initializer())

  # Run preliminary games to get antogonist started
  print('Preliminary games...')
  model.explore(game_count=20000)

  print('Serious games...')
  for antagonist in antagonists:
    env.add_opponent(antagonist)

  init_antagonists = []
  for copy_ops in antagonists_copy_ops:
    init_antagonists += copy_ops
  sess.run(init_antagonists)

  while True:
    model.explore(game_count=20000)
    EPOCH += 1

    print('Copying to antagonist #{}...'.format(ANTAGONIST_INDEX))
    sess.run(antagonists_copy_ops[ANTAGONIST_INDEX])
    antagonists[ANTAGONIST_INDEX].set_version(EPOCH)

    ANTAGONIST_INDEX = (ANTAGONIST_INDEX + 1) % len(antagonists)
