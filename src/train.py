import tensorflow as tf

from random_agent import RandomAgent
from env import Environment
from model import Model

e = Environment(RandomAgent())
model = Model(e)

state = e.reset()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  feed_dict = {}
  model.fill_feed_dict(feed_dict, [ state ])
  action = sess.run([ model.action ], feed_dict=feed_dict)
  print(action)
