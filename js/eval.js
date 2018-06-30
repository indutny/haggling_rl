'use strict';

const assert = require('assert');
const random = new (require('random-js'))();

const Generator = require('./generate').Generator;

const Neural = require('./agent');
const HalfOrAll = require('./half-or-all');

const ENABLE_LOG = false;
const TOTAL_MATCHES = 1000;

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
        return this.result(scene, previous);
      }

      previous = offer;
      offer = b.offer(this.inverseOffer(scene, offer));
      if (offer === undefined) {
        return this.result(scene, previous);
      }

      offer = this.inverseOffer(scene, offer);
    }
    return { accepted: false, a: 0, b: 0 };
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

  result(scene, offer) {
    const aOffer = offer;
    const bOffer = this.inverseOffer(scene, aOffer);

    return {
      accepted: true,
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

    sessions: 0,
    agreements: 0,
    score: 0,
  });
}

addContestant('neural', Neural);
addContestant('half-or-all', HalfOrAll);

for (let i = 0; i < TOTAL_MATCHES; i++) {
  const pair = random.sample(contestants, 2);

  const ab = arena.match(pair[0].agent, pair[1].agent);
  if (ab.accepted) {
    pair[0].agreements++;
    pair[1].agreements++;
  }

  pair[0].sessions++;
  pair[1].sessions++;
  pair[0].score += ab.a;
  pair[1].score += ab.b;
}

console.log(contestants.map((c) => {
  return {
    name: c.name,
    sessions: c.sessions,
    agreements: c.agreements,
    score: c.score,
  };
}));
