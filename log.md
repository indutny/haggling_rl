# Log of various configurations

## example.js

Baseline

    66.01% acceptance
    7.36 per accepted
    4.86 per deal

    "all": {
      "sessions": 762,
      "agreements": 503,
      "score": 3704
    },

## no-cons = -0.1 self, 0.0 remote (10 epoch)

PPO with all policies

    65.61% acceptance
    6.86 per accepted
    4.50 per deal

    "all": {
      "sessions": 887,
      "agreements": 582,
      "score": 3992
    },

## no-cons = -0.15 self, -0.15 remote

PPO with all policies, but lower self

## half-ppo, -0.15 no consensus

Trained on half-or-all policy.

### 3 epochs

    75.61% acceptance
    7.14 per accepted
    5.40 per deal

    "all": {
      "sessions": 652,
      "agreements": 493,
      "score": 3521
    },

### 18 epochs

    81.96% acceptance
    8.37 per accepted
    6.86 per deal

    "all": {
      "sessions": 887,
      "agreements": 727,
      "score": 6087
    },

### 22 epoch

    83.09% acceptance
    8.13 per accepted
    6.76 per deal

    "all": {
      "sessions": 887,
      "agreements": 737,
      "score": 5994
    },
