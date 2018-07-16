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
SAVE_EVERY = 100
BENCH_EVERY = 10

SWITCH_EVERY = 100

# Not really constants, but meh...
EPOCH = 0

env_list = []
bench_env = Environment()

bench_env.add_opponent(PolicyAgent(bench_env, policy='half_or_all'))

for i in range(CONCURRENCY):
  env = Environment()
  env_list.append(env)

writer = tf.summary.FileWriter(LOG_DIR)

with tf.Session() as sess:
  print('Initializing model')
  model = Model(CONFIG, env_list[0], sess, writer, name='haggle')

  writer.add_graph(tf.get_default_graph())
  saver = tf.train.Saver(max_to_keep=100, name=RUN_NAME)

  parallel = Model(CONFIG, env_list[0], sess, writer, name='haggle_parallel')

  sess.run(tf.global_variables_initializer())
  sess.graph.finalize()

  game_off = 0
  while True:
    if EPOCH % SWITCH_EVERY == 0:
      print('Switching agents')
      parallel, model = model, parallel

      for env in env_list:
        env.clear_opponents()
        env.add_opponent(PolicyAgent(env, policy='half_or_all'))
        env.add_opponent(PolicyAgent(env, policy='downsize'))
        env.add_opponent(parallel)

    EPOCH += 1
    print('Epoch {}'.format(EPOCH))

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
