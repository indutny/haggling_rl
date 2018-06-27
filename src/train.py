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

e = Environment(RandomAgent())

writer = tf.summary.FileWriter(LOG_DIR)

with tf.Session() as sess:
  model = Model(e, sess, writer)

  sess.run(tf.global_variables_initializer())

  model.explore()
