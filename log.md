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

# Self-play

## Relative, concurrency=8

### 10 epochs

    [ { name: 'neural',
        rounds: '1.5286',
        mean: '3.7750',
        meanAccepted: '5.0842',
        acceptance: '0.7425' },
      { name: 'half-or-all',
        rounds: '1.4309',
        mean: '5.9375',
        meanAccepted: '7.7236',
        acceptance: '0.7688' },
      { name: 'downsize',
        rounds: '1.5409',
        mean: '5.9600',
        meanAccepted: '8.2922',
        acceptance: '0.7188' } ]

### 20 epochs

    [ { name: 'neural',
        rounds: '1.8548',
        mean: '3.9463',
        meanAccepted: '5.2096',
        acceptance: '0.7575' },
      { name: 'half-or-all',
        rounds: '1.6534',
        mean: '6.2325',
        meanAccepted: '7.6472',
        acceptance: '0.8150' },
      { name: 'downsize',
        rounds: '1.8562',
        mean: '6.1437',
        meanAccepted: '8.0310',
        acceptance: '0.7650' } ]

### 30 epochs

    [ { name: 'neural',
        rounds: '1.6615',
        mean: '4.4350',
        meanAccepted: '5.5093',
        acceptance: '0.8050' },
      { name: 'half-or-all',
        rounds: '1.5856',
        mean: '6.1050',
        meanAccepted: '7.4679',
        acceptance: '0.8175' },
      { name: 'downsize',
        rounds: '1.7484',
        mean: '6.3037',
        meanAccepted: '8.0817',
        acceptance: '0.7800' } ]

### 40 epochs

    [ { name: 'neural',
        rounds: '1.6686',
        mean: '4.9813',
        meanAccepted: '5.8175',
        acceptance: '0.8562' },
      { name: 'half-or-all',
        rounds: '1.4605',
        mean: '5.9912',
        meanAccepted: '7.4310',
        acceptance: '0.8063' },
      { name: 'downsize',
        rounds: '1.7786',
        mean: '6.5338',
        meanAccepted: '8.0913',
        acceptance: '0.8075' } ]

### 50 epochs

    [ { name: 'neural',
        rounds: '1.6498',
        mean: '5.2363',
        meanAccepted: '6.2429',
        acceptance: '0.8387' },
      { name: 'half-or-all',
        rounds: '1.5202',
        mean: '5.9450',
        meanAccepted: '7.4081',
        acceptance: '0.8025' },
      { name: 'downsize',
        rounds: '1.7685',
        mean: '6.3575',
        meanAccepted: '8.0094',
        acceptance: '0.7937' } ]

### 60 epochs

    [ { name: 'neural',
        rounds: '1.6826',
        mean: '5.3487',
        meanAccepted: '6.4057',
        acceptance: '0.8350' },
      { name: 'half-or-all',
        rounds: '1.5842',
        mean: '5.9413',
        meanAccepted: '7.3462',
        acceptance: '0.8087' },
      { name: 'downsize',
        rounds: '1.8253',
        mean: '6.4538',
        meanAccepted: '7.9799',
        acceptance: '0.8087' } ]

### 70 epochs

    [ { name: 'neural',
        rounds: '1.7112',
        mean: '5.1338',
        meanAccepted: '6.3088',
        acceptance: '0.8137' },
      { name: 'half-or-all',
        rounds: '1.4509',
        mean: '5.8750',
        meanAccepted: '7.4367',
        acceptance: '0.7900' },
      { name: 'downsize',
        rounds: '1.8331',
        mean: '6.2763',
        meanAccepted: '7.9825',
        acceptance: '0.7863' } ]

### 80 rounds

    [ { name: 'neural',
        rounds: '1.6646',
        mean: '5.1350',
        meanAccepted: '6.4693',
        acceptance: '0.7937' },
      { name: 'half-or-all',
        rounds: '1.5039',
        mean: '5.9387',
        meanAccepted: '7.4119',
        acceptance: '0.8013' },
      { name: 'downsize',
        rounds: '1.8003',
        mean: '6.3137',
        meanAccepted: '7.9418',
        acceptance: '0.7950' } ]

### 90 rounds

    [ { name: 'neural',
        rounds: '1.6609',
        mean: '5.3150',
        meanAccepted: '6.7066',
        acceptance: '0.7925' },
      { name: 'half-or-all',
        rounds: '1.4547',
        mean: '5.8037',
        meanAccepted: '7.3816',
        acceptance: '0.7863' },
      { name: 'downsize',
        rounds: '1.7860',
        mean: '6.4775',
        meanAccepted: '8.0341',
        acceptance: '0.8063' } ]

### 100 rounds

    [ { name: 'neural',
        rounds: '1.7013',
        mean: '5.4188',
        meanAccepted: '7.0373',
        acceptance: '0.7700' },
      { name: 'half-or-all',
        rounds: '1.5183',
        mean: '5.4313',
        meanAccepted: '7.2417',
        acceptance: '0.7500' },
      { name: 'downsize',
        rounds: '1.8957',
        mean: '5.9363',
        meanAccepted: '7.8626',
        acceptance: '0.7550' } ]

### 110 rounds

    [ { name: 'neural',
        rounds: '1.6783',
        mean: '5.2100',
        meanAccepted: '7.2867',
        acceptance: '0.7150' },
      { name: 'half-or-all',
        rounds: '1.5034',
        mean: '5.2950',
        meanAccepted: '7.1554',
        acceptance: '0.7400' },
      { name: 'downsize',
        rounds: '1.7993',
        mean: '5.8987',
        meanAccepted: '7.8913',
        acceptance: '0.7475' } ]

### 120 rounds

    [ { name: 'neural',
        rounds: '1.6914',
        mean: '5.1842',
        meanAccepted: '7.2558',
        acceptance: '0.7145' },
      { name: 'half-or-all',
        rounds: '1.5167',
        mean: '5.4047',
        meanAccepted: '7.2304',
        acceptance: '0.7475' },
      { name: 'downsize',
        rounds: '1.8455',
        mean: '5.8483',
        meanAccepted: '7.8764',
        acceptance: '0.7425' } ]

### 130 rounds

    [ { name: 'neural',
        rounds: '1.5674',
        mean: '5.6128',
        meanAccepted: '7.1160',
        acceptance: '0.7887' },
      { name: 'half-or-all',
        rounds: '1.4502',
        mean: '5.7748',
        meanAccepted: '7.2524',
        acceptance: '0.7963' },
      { name: 'downsize',
        rounds: '1.7892',
        mean: '6.2092',
        meanAccepted: '7.9555',
        acceptance: '0.7805' } ]

### 140 rounds

    [ { name: 'neural',
        rounds: '1.5805',
        mean: '5.7477',
        meanAccepted: '7.2299',
        acceptance: '0.7950' },
      { name: 'half-or-all',
        rounds: '1.4072',
        mean: '5.6798',
        meanAccepted: '7.1714',
        acceptance: '0.7920' },
      { name: 'downsize',
        rounds: '1.7793',
        mean: '6.2092',
        meanAccepted: '7.9453',
        acceptance: '0.7815' } ]

### 150 rounds

    [ { name: 'neural',
        rounds: '1.4715',
        mean: '5.8513',
        meanAccepted: '6.9021',
        acceptance: '0.8478' },
      { name: 'half-or-all',
        rounds: '1.3565',
        mean: '5.9588',
        meanAccepted: '7.3069',
        acceptance: '0.8155' },
      { name: 'downsize',
        rounds: '1.7409',
        mean: '6.3957',
        meanAccepted: '7.9972',
        acceptance: '0.7997' } ]

### 160 rounds

    [ { name: 'neural',
        rounds: '1.4460',
        mean: '5.7780',
        meanAccepted: '6.8177',
        acceptance: '0.8475' },
      { name: 'half-or-all',
        rounds: '1.3632',
        mean: '5.9495',
        meanAccepted: '7.3135',
        acceptance: '0.8135' },
      { name: 'downsize',
        rounds: '1.7401',
        mean: '6.5105',
        meanAccepted: '7.9639',
        acceptance: '0.8175' } ]

### 170 rounds

    [ { name: 'neural',
        rounds: '1.6060',
        mean: '5.5280',
        meanAccepted: '7.4905',
        acceptance: '0.7380' },
      { name: 'half-or-all',
        rounds: '1.4608',
        mean: '5.4298',
        meanAccepted: '7.1798',
        acceptance: '0.7562' },
      { name: 'downsize',
        rounds: '1.8079',
        mean: '5.9443',
        meanAccepted: '7.9020',
        acceptance: '0.7522' } ]

### 180 rounds

    [ { name: 'neural',
        rounds: '1.5790',
        mean: '5.5555',
        meanAccepted: '7.5024',
        acceptance: '0.7405' },
      { name: 'half-or-all',
        rounds: '1.4335',
        mean: '5.5818',
        meanAccepted: '7.2116',
        acceptance: '0.7740' },
      { name: 'downsize',
        rounds: '1.7816',
        mean: '5.9633',
        meanAccepted: '7.9036',
        acceptance: '0.7545' } ]

### 190 rounds

    [ { name: 'neural',
        rounds: '1.5333',
        mean: '5.4490',
        meanAccepted: '7.6024',
        acceptance: '0.7167' },
      { name: 'half-or-all',
        rounds: '1.4192',
        mean: '5.4618',
        meanAccepted: '7.2461',
        acceptance: '0.7538' },
      { name: 'downsize',
        rounds: '1.7370',
        mean: '5.8025',
        meanAccepted: '7.9378',
        acceptance: '0.7310' } ]

### 200 rounds

    [ { name: 'neural',
        rounds: '1.6132',
        mean: '5.2332',
        meanAccepted: '7.6537',
        acceptance: '0.6837' },
      { name: 'half-or-all',
        rounds: '1.4481',
        mean: '5.3030',
        meanAccepted: '7.1421',
        acceptance: '0.7425' },
      { name: 'downsize',
        rounds: '1.8540',
        mean: '5.6292',
        meanAccepted: '7.8484',
        acceptance: '0.7173' } ]

### 220 rounds

    [ { name: 'neural',
        rounds: '1.5526',
        mean: '5.5210',
        meanAccepted: '7.4282',
        acceptance: '0.7432' },
      { name: 'half-or-all',
        rounds: '1.4372',
        mean: '5.4820',
        meanAccepted: '7.1497',
        acceptance: '0.7668' },
      { name: 'downsize',
        rounds: '1.7963',
        mean: '5.9050',
        meanAccepted: '7.8891',
        acceptance: '0.7485' } ]

## Relative, steps incentive

### 40 rounds

    [ { name: 'neural',
        rounds: '1.6335',
        mean: '4.7450',
        meanAccepted: '5.7446',
        acceptance: '0.8260' },
      { name: 'half-or-all',
        rounds: '1.4873',
        mean: '6.0652',
        meanAccepted: '7.4489',
        acceptance: '0.8143' },
      { name: 'downsize',
        rounds: '1.7772',
        mean: '6.3155',
        meanAccepted: '8.0735',
        acceptance: '0.7823' } ]
