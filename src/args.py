import argparse
import os

def parse_args(kind=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--pre', default='64')
  parser.add_argument('--lstm', type=int, default=128)
  parser.add_argument('--value_scale', type=float, default=0.5)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--grad_clip', type=float, default=0.5)
  parser.add_argument('--ppo', type=float, default=0.1)
  parser.add_argument('--ppo_epochs', type=int, default=10)
  parser.add_argument('--entropy', type=float, default=0.01)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--value_width', type=int, default=32)
  parser.add_argument('--tag')

  if kind == 'transform-save':
    parser.add_argument('source')
    parser.add_argument('target')
  elif kind == 'check-js':
    parser.add_argument('source')

  args = parser.parse_args()

  run_name = os.environ.get('HAGGLE_RUN')
  if run_name is None:
    run_name = 'p' + args.pre.replace(',', '_') + \
        '-lstm' + str(args.lstm) + \
        '-vs' + str(args.value_scale) + \
        '-lr' + str(args.lr) + \
        '-g' + str(args.grad_clip) + \
        '-ppo' + str(args.ppo) + \
        '-pe' + str(args.ppo_epochs) + \
        '-e' + str(args.entropy) + \
        '-g' + str(args.gamma) + \
        '-vw' + str(args.value_width) + \

  if not args.tag is None:
    run_name = str(args.tag) + '-' + run_name

  if args.pre == 'none':
    pre = []
  else:
    pre = [ int(v) for v in args.pre.split(',') ]

  config = {
    'pre': pre,
    'lstm': args.lstm,
    'value_scale': args.value_scale,
    'lr': args.lr,
    'grad_clip': args.grad_clip,
    'ppo': args.ppo,
    'ppo_epochs': args.ppo_epochs,
    'entropy': args.entropy,
    'gamma': args.gamma,
    'value_width': args.value_width,
  }

  return run_name, config, args
