import sys
import tensorflow as tf

from env import Environment
from policy_agent import PolicyAgent
from model import Model

env = Environment()

env.add_opponent(PolicyAgent(policy='downsize'))

with tf.Session() as sess:
  model = Model(env, sess, None, name='haggle')
  saver = tf.train.Saver(max_to_keep=10000, name='test')

  saver.restore(sess, sys.argv[1])
  print(sess.run(model.action, feed_dict={
    model.input: [ [
      1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0,
      0, 0, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0
    ] ],
    model.rnn_state: [ model.initial_state ],
  }))
