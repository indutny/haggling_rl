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

### 22 epochs

    83.09% acceptance
    8.13 per accepted
    6.76 per deal

    "all": {
      "sessions": 887,
      "agreements": 737,
      "score": 5994
    },

### 25 epochs

    82.19% acceptance
    8.52 per accepted
    7.00 per deal

    "all": {
      "sessions": 887,
      "agreements": 729,
      "score": 6214
    },

## downsize-ppo

Trained against just downsize.

### 3 epochs

    71.22% acceptance
    7.70 per accepted
    5.48 per deal

    "all": {
      "sessions": 476,
      "agreements": 339,
      "score": 2609
    },

### 7 epochs

    77.90% acceptance
    8.33 per accepted
    6.49 per deal

    "all": {
      "sessions": 887,
      "agreements": 691,
      "score": 5757
    },

## downsize with single adv and flat reward

### 10 epochs

    75.53% acceptance
    7.91 per accepted
    5.97 per deal

    "all": {
      "sessions": 887,
      "agreements": 670,
      "score": 5303
    },

## anneal

### 5 epochs

    "all": {
      "sessions": 909,
      "agreements": 595,
      "score": 4161
    },


## masked

### 3 epochs

    "all": {
      "sessions": 1509,
      "agreements": 1062,
      "score": 8068
    },

### 6 epochs

    "all": {
      "sessions": 1699,
      "agreements": 1126,
      "score": 9071
    },

### 10 epochs

    "all": {
      "sessions": 4285,
      "agreements": 2723,
      "score": 22414
    },

### 24 epochs

    "all": {
      "sessions": 2066,
      "agreements": 1195,
      "score": 10031
    },
