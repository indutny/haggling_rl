import argparse
import os

def parse_args(kind=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--pre', default='64')
  parser.add_argument('--lstm', type=int, default=128)
  parser.add_argument('--value_scale', type=float, default=0.5)
  parser.add_argument('--max_steps', type=int, default=50)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--grad_clip', type=float, default=0.5)
  parser.add_argument('--ppo', type=float, default=0.1)

  if kind == 'transform-save':
    parser.add_argument('source')
    parser.add_argument('target')

  args = parser.parse_args()

  run_name = os.environ.get('HAGGLE_RUN')
  if run_name is None:
    run_name = 'p' + args.pre.replace(',', '_') + \
        '-lstm' + str(args.lstm) + \
        '-vs' + str(args.value_scale) + \
        '-ms' + str(args.max_steps) + \
        '-lr' + str(args.lr) + \
        '-g' + str(args.grad_clip) + \
        '-ppo' + str(args.ppo)

  if args.pre == 'none':
    pre = []
  else:
    pre = [ int(v) for v in args.pre.split(',') ]

  config = {
    'pre': pre,
    'lstm': args.lstm,
    'value_scale': args.value_scale,
    'max_steps': args.max_steps,
    'lr': args.lr,
    'grad_clip': args.grad_clip,
    'ppo': args.ppo,
  }

  return run_name, config, args