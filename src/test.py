from env import Environment
from policy_agent import PolicyAgent

env = Environment()

env.add_opponent(PolicyAgent(policy='downsize'))

env.reset()
print(env._make_state())
