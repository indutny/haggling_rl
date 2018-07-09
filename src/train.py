import os
import time
import random
import tensorflow as tf

from policy_agent import PolicyAgent
from env import Environment
from model import Model
from args import parse_args

RUN_NAME, CONFIG, _ = parse_args()
print('Booting up {}'.format(RUN_NAME))
print('config', CONFIG)

LOG_DIR = os.path.join('.', 'logs', RUN_NAME)
SAVE_DIR = os.path.join('.', 'saves', RUN_NAME)

CONCURRENCY = 32
SAVE_EVERY = 10
BENCH_EVERY = 1

MAX_ANTAGONISTS = 500
NUM_ANTAGONISTS = 32
ANTAGONIST_EPOCH = 1
ANTAGONIST_UPDATE_EVERY = 1
ANTAGONISTS = []
ANTAGONIST_WEIGHTS = []

# Not really constants, but meh...
EPOCH = 0

env_list = []
bench_env = Environment()

bench_env.add_opponent(PolicyAgent(bench_env, policy='downsize'))

for i in range(CONCURRENCY):
  env = Environment()

  # env.add_opponent(PolicyAgent(env, policy='half_or_all'))
  # env.add_opponent(PolicyAgent(env, policy='downsize'))
  # env.add_opponent(PolicyAgent(env, policy='altruist'))
  # env.add_opponent(PolicyAgent(env, policy='greedy'))
  # env.add_opponent(PolicyAgent(env, policy='stubborn'))

  env_list.append(env)

writer = tf.summary.FileWriter(LOG_DIR)

with tf.Session() as sess:
  model = Model(CONFIG, env_list[0], sess, writer, name='haggle')

  writer.add_graph(tf.get_default_graph())
  saver = tf.train.Saver(max_to_keep=10000, name=RUN_NAME)

  for i in range(NUM_ANTAGONISTS):
    antagonist = Model(CONFIG, env, sess, None, name='antagonist_{}'.format(i))
    ANTAGONISTS.append(antagonist)

  sess.run(tf.global_variables_initializer())
  sess.graph.finalize()

  game_off = 0
  while True:
    EPOCH += 1
    print('Epoch {}'.format(EPOCH))

    if EPOCH == ANTAGONIST_EPOCH:
      print('Adding antagonists!')
      for env in env_list:
        env.clear_opponents()
        for antagonist in ANTAGONISTS:
          env.add_opponent(antagonist)

      # Update weights
      print('Initializing their weights...')
      weights = model.save_weights(sess)
      ops = []
      feed_dict = {}
      for antagonist in ANTAGONISTS:
        a_dict, a_ops = antagonist.load_weights(weights)

        feed_dict.update(a_dict)
        ops += a_ops
      sess.run(ops, feed_dict=feed_dict)

      print('Time for real games!')

    model.explore(env_list, game_count=1024, game_off=game_off)
    game_off += 1024

    if EPOCH % SAVE_EVERY == 0:
      saver.save(sess, os.path.join(SAVE_DIR, '{:08d}'.format(EPOCH)))

    if EPOCH % BENCH_EVERY == 0:
      print('Running benchmark...')
      bench = bench_env.bench(model)

      # TODO(indutny): move logging to model?
      summary = tf.Summary()
      for key, value in bench.items():
        summary.value.add(tag='bench/{}'.format(key), simple_value=value)
      writer.add_summary(summary, model.writer_step)

    if EPOCH < ANTAGONIST_EPOCH:
      continue

    if EPOCH % ANTAGONIST_UPDATE_EVERY == 0:
      print('Adding new antagonist to the pool')
      weights = model.save_weights(sess)
      ANTAGONIST_WEIGHTS.append({
        'epoch': EPOCH,
        'weights': weights
      })

      # Equal spacing between everyone
      if len(ANTAGONIST_WEIGHTS) >= 2 * MAX_ANTAGONISTS:
        ANTAGONIST_WEIGHTS = ANTAGONIST_WEIGHTS[::2]
        ANTAGONIST_UPDATE_EVERY *= 2

    if len(ANTAGONIST_WEIGHTS) > 0:
      if len(ANTAGONIST_WEIGHTS) < NUM_ANTAGONISTS:
        weights = ANTAGONIST_WEIGHTS
      else:
        weights = random.sample(ANTAGONIST_WEIGHTS, NUM_ANTAGONISTS)

      env.clear_opponents()
      ops = []
      feed_dict = {}
      versions = []
      for antagonist, save in zip(ANTAGONISTS, weights):
        versions.append(save['epoch'])
        a_dict, a_ops = antagonist.load_weights(save['weights'])

        feed_dict.update(a_dict)
        ops += a_ops

        antagonist.set_version(save['epoch'])
        env.add_opponent(antagonist)

      sess.run(ops, feed_dict=feed_dict)
      print('Loaded {} into antagonists'.format(versions))
