'use strict'; /*jslint node:true*/

const random = new (require('random-js'))();

const MIN_OFFER = 0.5;

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
    this.fillOffers(this.counts.slice().fill(0), 0);

    this.possibleValues = this.possibleValues.filter((v) => {
      return !v.every((cell, i) => {
        return this.values[i] === cell;
      });
    });
    this.possibleOffers = this.possibleOffers.filter((o) => {
      return this.offerValue(o, this.values) >= MIN_OFFER;
    });

    this.crossMap = this.possibleOffers.map((offer) => {
      return {
        offer,
        values: this.possibleValues.map((values) => {
          return {
            self: this.offerValue(offer, this.values),
            opponent: this.offerValue(this.invertOffer(offer), values),
          };
        }),
      };
    });

    this.pastOffers = [];
    this.used = this.possibleOffers.map(_ => false);
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

    const scores = this.crossMap.map((entry) => {
      return entry.values.reduce((acc, current, i) => {
        const estimate = estimates[i];

        // Unlikely to be accepted
        if (current.opponent < 0.5) {
          return acc;
        }

        const delta = current.self - current.opponent;

        return acc + estimate * Math.max(delta + 0.1, 0);
      }, 0);
    });

    let max = -Infinity;
    let maxI = null;

    for (let i = 0; i < scores.length; i++) {
      if (!this.used[i] && scores[i] > max) {
        max = scores[i];
        maxI = i;
      }
    }

    // Can't find right offer
    if (maxI === null) {
      return this.counts;
    }

    this.used[maxI] = true;

    return this.possibleOffers[maxI];
  }

  estimate(pastOffers) {
    const scores = [];
    for (const values of this.possibleValues) {
      let score = 0;
      for (const o of pastOffers) {
        score += Math.max(0, this.offerValue(o, values) - 0.5);
      }
      scores.push(score);
    }
    return scores;
  }
};
