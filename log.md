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

## Incentive runs (half-or-all and downsize)

### 10 rounds

    [ { name: 'neural',
        rounds: '1.6067',
        mean: '3.7942',
        meanAccepted: '4.7637',
        acceptance: '0.7965' },
      { name: 'half-or-all',
        rounds: '1.5371',
        mean: '6.2780',
        meanAccepted: '7.6772',
        acceptance: '0.8177' },
      { name: 'downsize',
        rounds: '1.7016',
        mean: '6.2260',
        meanAccepted: '8.1572',
        acceptance: '0.7632' } ]

### 20 rounds

    [ { name: 'neural',
        rounds: '1.8146',
        mean: '4.5440',
        meanAccepted: '6.3464',
        acceptance: '0.7160' },
      { name: 'half-or-all',
        rounds: '1.6313',
        mean: '5.8465',
        meanAccepted: '7.4596',
        acceptance: '0.7837' },
      { name: 'downsize',
        rounds: '1.8474',
        mean: '5.7542',
        meanAccepted: '7.9671',
        acceptance: '0.7222' } ]

### 30 rounds

    [ { name: 'neural',
        rounds: '1.7936',
        mean: '4.8303',
        meanAccepted: '6.8056',
        acceptance: '0.7097' },
      { name: 'half-or-all',
        rounds: '1.5350',
        mean: '5.7752',
        meanAccepted: '7.4137',
        acceptance: '0.7790' },
      { name: 'downsize',
        rounds: '1.8205',
        mean: '5.7195',
        meanAccepted: '7.9742',
        acceptance: '0.7173' } ]

### 40 rounds

    [ { name: 'neural',
        rounds: '1.8012',
        mean: '5.2257',
        meanAccepted: '6.8805',
        acceptance: '0.7595' },
      { name: 'half-or-all',
        rounds: '1.5735',
        mean: '5.7497',
        meanAccepted: '7.3315',
        acceptance: '0.7843' },
      { name: 'downsize',
        rounds: '1.8526',
        mean: '5.9915',
        meanAccepted: '7.9542',
        acceptance: '0.7532' } ]

### 50 rounds

    [ { name: 'neural',
        rounds: '1.8809',
        mean: '5.0853',
        meanAccepted: '7.1023',
        acceptance: '0.7160' },
      { name: 'half-or-all',
        rounds: '1.6228',
        mean: '5.6262',
        meanAccepted: '7.3116',
        acceptance: '0.7695' },
      { name: 'downsize',
        rounds: '1.9468',
        mean: '5.7317',
        meanAccepted: '7.8196',
        acceptance: '0.7330' } ]

### 60 rounds

    [ { name: 'neural',
        rounds: '1.7742',
        mean: '5.4230',
        meanAccepted: '7.2670',
        acceptance: '0.7462' },
      { name: 'half-or-all',
        rounds: '1.5205',
        mean: '5.6528',
        meanAccepted: '7.2564',
        acceptance: '0.7790' },
      { name: 'downsize',
        rounds: '1.8588',
        mean: '5.8715',
        meanAccepted: '7.8945',
        acceptance: '0.7438' } ]

### 70 rounds

    [ { name: 'neural',
        rounds: '1.9310',
        mean: '5.5172',
        meanAccepted: '7.3222',
        acceptance: '0.7535' },
      { name: 'half-or-all',
        rounds: '1.6050',
        mean: '5.6890',
        meanAccepted: '7.2195',
        acceptance: '0.7880' },
      { name: 'downsize',
        rounds: '1.9613',
        mean: '5.8063',
        meanAccepted: '7.8146',
        acceptance: '0.7430' } ]

### 80 rounds

    [ { name: 'neural',
        rounds: '1.9046',
        mean: '5.5323',
        meanAccepted: '7.3837',
        acceptance: '0.7492' },
      { name: 'half-or-all',
        rounds: '1.5762',
        mean: '5.5522',
        meanAccepted: '7.1967',
        acceptance: '0.7715' },
      { name: 'downsize',
        rounds: '1.9703',
        mean: '5.7975',
        meanAccepted: '7.8160',
        acceptance: '0.7418' } ]

### 90 rounds

    [ { name: 'neural',
        rounds: '1.9714',
        mean: '5.7732',
        meanAccepted: '7.4929',
        acceptance: '0.7705' },
      { name: 'half-or-all',
        rounds: '1.5857',
        mean: '5.6400',
        meanAccepted: '7.1779',
        acceptance: '0.7857' },
      { name: 'downsize',
        rounds: '1.9781',
        mean: '6.0652',
        meanAccepted: '7.8186',
        acceptance: '0.7758' } ]

### 120 rounds

    [ { name: 'neural',
        rounds: '2.0027',
        mean: '5.7320',
        meanAccepted: '7.6811',
        acceptance: '0.7462' },
      { name: 'half-or-all',
        rounds: '1.6305',
        mean: '5.4875',
        meanAccepted: '7.1082',
        acceptance: '0.7720' },
      { name: 'downsize',
        rounds: '2.0288',
        mean: '5.8110',
        meanAccepted: '7.7043',
        acceptance: '0.7542' } ]

### 150 rounds

    [ { name: 'neural',
        rounds: '1.9726',
        mean: '5.9523',
        meanAccepted: '7.5801',
        acceptance: '0.7853' },
      { name: 'half-or-all',
        rounds: '1.5903',
        mean: '5.5617',
        meanAccepted: '7.0828',
        acceptance: '0.7853' },
      { name: 'downsize',
        rounds: '2.0391',
        mean: '5.8370',
        meanAccepted: '7.6702',
        acceptance: '0.7610' } ]

### 200 rounds

    [ { name: 'neural',
        rounds: '2.0153',
        mean: '6.1730',
        meanAccepted: '7.7283',
        acceptance: '0.7987' },
      { name: 'half-or-all',
        rounds: '1.5904',
        mean: '5.5207',
        meanAccepted: '7.0508',
        acceptance: '0.7830' },
      { name: 'downsize',
        rounds: '2.0963',
        mean: '5.9328',
        meanAccepted: '7.6183',
        acceptance: '0.7788' } ]

### 270 rounds

    [ { name: 'neural',
        rounds: '2.0412',
        mean: '6.1430',
        meanAccepted: '7.7932',
        acceptance: '0.7883' },
      { name: 'half-or-all',
        rounds: '1.5581',
        mean: '5.5533',
        meanAccepted: '7.0006',
        acceptance: '0.7933' },
      { name: 'downsize',
        rounds: '2.1188',
        mean: '5.9200',
        meanAccepted: '7.6044',
        acceptance: '0.7785' } ]

### 310 rounds

    [ { name: 'neural',
        rounds: '1.9821',
        mean: '6.3063',
        meanAccepted: '7.7975',
        acceptance: '0.8087' },
      { name: 'half-or-all',
        rounds: '1.5480',
        mean: '5.6345',
        meanAccepted: '6.9864',
        acceptance: '0.8065' },
      { name: 'downsize',
        rounds: '2.1153',
        mean: '5.9372',
        meanAccepted: '7.5658',
        acceptance: '0.7847' } ]

### 370 rounds

    [ { name: 'neural',
        rounds: '1.9802',
        mean: '6.2705',
        meanAccepted: '7.7677',
        acceptance: '0.8073' },
      { name: 'half-or-all',
        rounds: '1.5054',
        mean: '5.6505',
        meanAccepted: '7.0171',
        acceptance: '0.8053' },
      { name: 'downsize',
        rounds: '2.1274',
        mean: '5.8545',
        meanAccepted: '7.5542',
        acceptance: '0.7750' } ]

### 400 rounds

    [ { name: 'neural',
        rounds: '2.0311',
        mean: '6.1075',
        meanAccepted: '7.8276',
        acceptance: '0.7802' },
      { name: 'half-or-all',
        rounds: '1.5676',
        mean: '5.3948',
        meanAccepted: '6.9009',
        acceptance: '0.7817' },
      { name: 'downsize',
        rounds: '2.1899',
        mean: '5.7038',
        meanAccepted: '7.5099',
        acceptance: '0.7595' } ]

### 430 rounds

    [ { name: 'neural',
        rounds: '2.0363',
        mean: '6.4055',
        meanAccepted: '7.6943',
        acceptance: '0.8325' },
      { name: 'half-or-all',
        rounds: '1.5757',
        mean: '5.6273',
        meanAccepted: '6.8856',
        acceptance: '0.8173' },
      { name: 'downsize',
        rounds: '2.1827',
        mean: '5.9640',
        meanAccepted: '7.5137',
        acceptance: '0.7937' } ]

### 490 rounds

    [ { name: 'neural',
        rounds: '2.0473',
        mean: '6.2625',
        meanAccepted: '7.6395',
        acceptance: '0.8197' },
      { name: 'half-or-all',
        rounds: '1.5142',
        mean: '5.6123',
        meanAccepted: '6.9330',
        acceptance: '0.8095' },
      { name: 'downsize',
        rounds: '2.1166',
        mean: '5.9880',
        meanAccepted: '7.5918',
        acceptance: '0.7887' } ]

## antagonist on 20

### 2 epochs

    [ { name: 'neural',
        rounds: '1.6399',
        mean: '3.4407',
        meanAccepted: '5.5631',
        acceptance: '0.6185' },
      { name: 'half-or-all',
        rounds: '1.5227',
        mean: '5.4120',
        meanAccepted: '7.5141',
        acceptance: '0.7202' },
      { name: 'downsize',
        rounds: '1.7510',
        mean: '5.4250',
        meanAccepted: '8.0163',
        acceptance: '0.6767' } ]

### 4 epochs

    [ { name: 'neural',
        rounds: '1.7791',
        mean: '4.8208',
        meanAccepted: '6.9841',
        acceptance: '0.6903' },
      { name: 'half-or-all',
        rounds: '1.5733',
        mean: '5.5693',
        meanAccepted: '7.3183',
        acceptance: '0.7610' },
      { name: 'downsize',
        rounds: '1.8859',
        mean: '5.5432',
        meanAccepted: '7.8823',
        acceptance: '0.7033' } ]

### 8 epochs

    [ { name: 'neural',
        rounds: '1.8871',
        mean: '4.8208',
        meanAccepted: '8.0012',
        acceptance: '0.6025' },
      { name: 'half-or-all',
        rounds: '1.5749',
        mean: '5.0400',
        meanAccepted: '7.1718',
        acceptance: '0.7027' },
      { name: 'downsize',
        rounds: '1.8878',
        mean: '5.2555',
        meanAccepted: '7.8353',
        acceptance: '0.6707' } ]
