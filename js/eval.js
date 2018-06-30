'use strict';

const assert = require('assert');
const random = new (require('random-js'))();

const Generator = require('./generate').Generator;

const Neural = require('./agents/neural');
const HalfOrAll = require('./agents/half-or-all');
const Downsize = require('./agents/downsize');

const ENABLE_LOG = false;
const TOTAL_MATCHES = 200;

function log(msg) {
  if (ENABLE_LOG) {
    console.error(msg);
  }
}

class Arena {
  constructor(types, minObj, maxObj, total, maxRounds) {
    this.sets = new Generator(types, minObj, maxObj, total, maxRounds);
  }

  match(A, B) {
    const scene = this.sets.get(random);

    const a = new A('a', scene.counts, scene.valuations[0], scene.max_rounds,
        log);
    const b = new B('b', scene.counts, scene.valuations[1], scene.max_rounds,
        log);

    let offer = undefined;
    for (let i = 0; i < scene.max_rounds; i++) {
      let previous = offer;

      offer = a.offer(offer);
      if (offer === undefined) {
        assert.notStrictEqual(previous, undefined, 'Invalid first offer');

        // Accept
        return this.result(scene, i, previous);
      }

      previous = offer;
      offer = b.offer(this.inverseOffer(scene, offer));
      if (offer === undefined) {
        return this.result(scene, i, previous);
      }

      offer = this.inverseOffer(scene, offer);
    }
    return { accepted: false, rounds: scene.max_rounds, a: 0, b: 0 };
  }

  offerValue(scene, player, offer) {
    let value = 0;
    for (let i = 0; i < scene.valuations[player].length; i++) {
      value += (scene.valuations[player][i] * offer[i]) | 0;
    }
    return value;
  }

  inverseOffer(scene, offer) {
    const res = offer.slice();
    for (let i = 0; i < scene.counts.length; i++) {
      res[i] = (scene.counts[i] - res[i]) | 0;
    }
    return res;
  }

  result(scene, rounds, offer) {
    const aOffer = offer;
    const bOffer = this.inverseOffer(scene, aOffer);

    return {
      accepted: true,
      rounds,
      a: this.offerValue(scene, 0, aOffer),
      b: this.offerValue(scene, 1, bOffer),
    };
  }
}

const arena = new Arena(3, 1, 6, 10, 5);

const contestants = [];

function addContestant(name, A) {
  contestants.push({
    agent: A,
    name,

    rounds: 0,
    sessions: 0,
    agreements: 0,
    score: 0,
  });
}

addContestant('neural', Neural);
addContestant('half-or-all', HalfOrAll);
addContestant('downsize', Downsize);

const pairs = [];
for (const a of contestants) {
  for (const b of contestants) {
    if (a !== b) {
      pairs.push({ a, b });
    }
  }
}

for (let i = 0; i < TOTAL_MATCHES; i++) {
  for (const pair of pairs) {
    const ab = arena.match(pair.a.agent, pair.b.agent);
    if (ab.accepted) {
      pair.a.agreements++;
      pair.b.agreements++;
    }

    pair.a.sessions++;
    pair.b.sessions++;
    pair.a.rounds += ab.rounds;
    pair.b.rounds += ab.rounds;
    pair.a.score += ab.a;
    pair.b.score += ab.b;
  }
}

console.log(contestants.map((c) => {
  return {
    name: c.name,
    rounds: (c.rounds / c.sessions).toFixed(4),
    mean: (c.score / c.sessions).toFixed(4),
    meanAccepted: (c.score / c.agreements).toFixed(4),
    acceptance: (c.agreements / c.sessions).toFixed(4),
  };
}));