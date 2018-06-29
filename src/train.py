import os
import time
import tensorflow as tf

from policy_agent import PolicyAgent
from env import Environment
from model import Model

RUN_NAME = os.environ.get('HAGGLE_RUN')
if RUN_NAME is None:
  RUN_NAME = time.strftime('%d.%m.%Y_%H%M%S')
LOG_DIR = os.path.join('.', 'logs', RUN_NAME)
SAVE_DIR = os.path.join('.', 'saves', RUN_NAME)

NUM_ANTAGONISTS = 1
ANTAGONIST_UPDATE_FREQ = 1

# Not really constants, but meh...
ANTAGONIST_INDEX = 0
EPOCH = 0

env = Environment()

env.add_opponent(PolicyAgent(policy='half_or_all'))
env.add_opponent(PolicyAgent(policy='downsize'))

writer = tf.summary.FileWriter(LOG_DIR)

def entropy(game_count):
  if game_count < 100000:
    return 0.01
  else:
    return 0.01 * (0.9999 ** (game_count - 100000))

with tf.Session() as sess:
  model = Model(env, sess, writer, name='haggle')
  saver = tf.train.Saver(max_to_keep=10000, name=RUN_NAME)

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
  game_off = 0
  model.explore(game_count=20000, game_off=game_off, entropy_scale=entropy)
  game_off += 20000

  print('Serious games...')
  for antagonist in antagonists:
    env.add_opponent(antagonist)

  init_antagonists = []
  for copy_ops in antagonists_copy_ops:
    init_antagonists += copy_ops
  sess.run(init_antagonists)

  while True:
    print('Running...')
    model.explore(game_count=20000, game_off=game_off, entropy_scale=entropy)
    game_off += 20000
    EPOCH += 1

    print('Saving...')
    saver.save(sess, os.path.join(SAVE_DIR, '{:08d}'.format(EPOCH)))

    if EPOCH % ANTAGONIST_UPDATE_FREQ == 0 and NUM_ANTAGONISTS != 0:
      print('Copying to antagonist #{}...'.format(ANTAGONIST_INDEX))
      sess.run(antagonists_copy_ops[ANTAGONIST_INDEX])
      antagonists[ANTAGONIST_INDEX].set_version(EPOCH)

      ANTAGONIST_INDEX = (ANTAGONIST_INDEX + 1) % len(antagonists)
