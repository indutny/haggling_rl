'use strict'; /*jslint node:true*/

const random = new (require('random-js'))();

const MIN_OFFER = 0.5;
const SAMPLE = 1000;

function softmax(x) {
  const res = x.slice();
  let sum = 0;
  for (let i = 0; i < res.length; i++) {
    const t = Math.exp(res[i]);
    sum += t;
    res[i] = t;
  }

  for (let i = 0; i < res.length; i++) {
    res[i] /= sum;
  }
  return res;
}

function sampleRandom(probs) {
  let roll = Math.random();
  let res = 0;
  for (;;) {
    roll -= probs[res];
    if (roll <= 0) {
      break;
    }
    res++;
  }
  return res;
}

module.exports = class Estimator {
  constructor(me, counts, values, maxRounds, log){
    this.counts = counts;
    this.values = values;
    this.maxRounds = maxRounds;

    this.round = 0;

    this.total = 0;
    for (let i = 0 ; i < this.counts.length; i++) {
      this.total += this.counts[i] * this.values[i];
    }

    this.possibleValues = [];
    this.possibleOffers = [];

    this.fillValues(this.counts.slice().fill(0), 0, 0);
    this.fillOffers(this.counts.slice().fill(0), 0, 0);

    this.possibleValues = this.possibleValues.filter((v) => {
      return !v.every((cell, i) => {
        return this.values[i] === cell;
      });
    });
    this.possibleOffers = this.possibleOffers.filter((o) => {
      return this.offerValue(o, this.values) >= MIN_OFFER;
    });

    this.crossMap = this.possibleValues.map((values) => {
      return this.possibleOffers.map((offer) => {
        return {
          offer,
          self: this.offerValue(offer, this.values),
          opponent: this.offerValue(this.invertOffer(offer), values),
        };
      });
    });

    this.pastOffers = [];
  }

  invertOffer(offer) {
    return offer.map((count, i) => this.counts[i] - count);
  }

  offerValue(offer, values) {
    let res = 0;
    for (let i = 0; i < offer.length; i++) {
      res += offer[i] * values[i];
    }
    return res / this.total;
  }

  fillValues(values, i, total) {
    const count = this.counts[i];
    const max = (this.total - total) / count | 0;
    if (i === this.counts.length - 1) {
      if (total + max * count === this.total) {
        values[i] = max;
        this.possibleValues.push(values.slice());
      }
      return;
    }
    for (let j = 0; j <= max; j++) {
      values[i] = j;
      this.fillValues(values, i + 1, total + j * count);
    }
  }

  fillOffers(offer, i) {
    if (i === this.counts.length) {
      this.possibleOffers.push(offer.slice());
      return;
    }

    for (let j = 0; j <= this.counts[i]; j++) {
      offer[i] = j;
      this.fillOffers(offer, i + 1);
    }
  }

  offer(o) {
    this.round++;

    // Ask everything
    if (o === undefined) {
      return this.counts;
    }

    this.pastOffers.push(this.invertOffer(o));
    const estimates = this.estimate(this.pastOffers);

    // Raise the diff between probabilities a bit
    const probs = softmax(estimates.map(e => e * 2));

    // Get opponent values
    const opponentI = sampleRandom(probs);

    // Find optimal offer
    const optimal = this.crossMap[opponentI].filter((entry) => {
      return entry.self >= MIN_OFFER && entry.opponent >= MIN_OFFER;
    }).map((entry) => {
      return {
        offer: entry.offer,
        delta: entry.self,
      };
    });

    const offerProbs = softmax(optimal.map(e => e.delta));
    const proposedI = sampleRandom(offerProbs);

    const proposed = optimal[proposedI].offer;
    this.pastOffers.push(this.invertOffer(proposed));

    if (this.offerValue(proposed, this.values) ===
        this.offerValue(o, this.values)) {
      // Accept
      return undefined;
    }

    return proposed;
  }

  estimate(pastOffers) {
    const scores = [];
    for (const values of this.possibleValues) {
      let score = 0;
      for (const o of pastOffers) {
        score += this.offerValue(o, values);
      }
      scores.push(score);
    }
    return scores;
  }
};
