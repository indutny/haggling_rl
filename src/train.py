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

# Not really constants, but meh...
EPOCH = 0

env = Environment()
opponent_env = Environment()

writer = tf.summary.FileWriter(LOG_DIR)

with tf.Session() as sess:
  model = Model(env, sess, writer, name='haggle')
  saver = tf.train.Saver(max_to_keep=10000, name=RUN_NAME)

  opponent = Model(opponent_env, sess, writer, name='opponent')

  env.add_opponent(opponent)
  opponent_env.add_opponent(model)

  sess.run(tf.global_variables_initializer())
  sess.graph.finalize()

  game_off = 0
  while True:
    EPOCH += 1
    print('Epoch {}'.format(EPOCH))

    print('Model training')
    model.explore(game_count=1024, game_off=game_off, prefix='model')

    print('Opponent training')
    opponent.explore(game_count=1024, game_off=game_off, prefix='opponent')
    game_off += 1024

    if EPOCH % SAVE_EVERY == 0:
      saver.save(sess, os.path.join(SAVE_DIR, '{:08d}'.format(EPOCH)))
