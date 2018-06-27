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

env = Environment()

env.add_opponent(RandomAgent())

writer = tf.summary.FileWriter(LOG_DIR)

with tf.Session() as sess:
  model = Model(env, sess, writer, name='haggle')

  antagonist = Model(env, sess, writer, name='haggle_antagonist')
  copy_ops = antagonist.copy_from(model)

  sess.run(tf.global_variables_initializer())

  # Run preliminary games to get antogonist started
  print('Preliminary games...')
  model.explore(game_count=20000)

  print('Serious games...')
  env.clear_opponents()
  env.add_opponent(antagonist)

  while True:
    print('Copying to antagonist...')
    sess.run(copy_ops)
    antagonist.increment_version()

    model.explore(game_count=20000)
