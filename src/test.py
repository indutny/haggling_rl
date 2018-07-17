from env import Environment
from policy_agent import PolicyAgent

env = Environment()

env.add_opponent(PolicyAgent(env, policy='estimator'))

print(env.bench(PolicyAgent(env, policy='downsize')))

env.reset()
print(env._make_state())
