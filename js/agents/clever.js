'use strict'; /*jslint node:true*/

module.exports = class Clever {
  constructor(me, counts, values, maxRounds, log){
    this.me = me;
    this.counts = counts;
    this.values = values;
    this.round = 0;
    this.maxRounds = maxRounds;
    this.log = log;

    // Log total value
    this.total = 0;
    for (let i = 0 ; i < this.counts.length; i++) {
      this.total += this.counts[i] * this.values[i];
    }

    this.possibleValues = [];
    this.possibleOffers = [];

    this.fillOffers(this.counts.slice().fill(0), 0);
    this.fillValues(this.counts.slice().fill(0), 0, 0);

    this.possibleValues = this.possibleValues.filter((v) => {
      return !v.every((cell, i) => {
        return this.values[i] === cell;
      });
    });

    this.offerValues =
        this.possibleOffers.map(o => this.offerValue(o, this.values));

    this.history = [];
  }

  invertOffer(offer) {
    return offer.map((count, i) => this.counts[i] - count);
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

  offer(o) {
    if (o !== undefined) {
      // Count items that opponent cares most about
      this.history.push(this.invertOffer(o));

      // Accept good offers
      if (this.offerValue(o, this.values) >= 0.8) {
        return undefined;
      }
    }

    // We don't know much yet, so it is better to return some greedy offer
    this.round++;
    if (this.round < this.maxRounds) {
      return this.greedyOffer();
    }

    // Return clever offer using collected information
    return this.cleverOffer();
  }

  offerValue(offer, values) {
    let res = 0;
    for (let i = 0; i < offer.length; i++) {
      res += offer[i] * values[i];
    }
    return res / this.total;
  }

  greedyOffer() {
    const greedyOffers = this.possibleOffers.filter((_, i) => {
      return this.offerValues[i] >= 0.8;
    });

    if (greedyOffers.length === 0) {
      return this.counts;
    }

    return greedyOffers[(Math.random() * greedyOffers.length) | 0];
  }

  cleverOffer() {
    let offers;
    let searchValue = 0.6;
    do {
      offers = this.possibleOffers.filter((_, i) => {
        return this.offerValues[i] === searchValue;
      });
      searchValue += 0.1;
    } while (offers.length === 0 && searchValue <= 1);

    if (offers.length === 0) {
      this.log('no suitable offer');
      return this.counts;
    }

    this.log('search_value=' + (searchValue - 0.1));

    const valueScores = this.possibleValues.map((values) => {
      return this.history.reduce((acc, offer) => {
        const opponentValue = this.offerValue(offer, values);
        const selfValue = this.offerValue(this.invertOffer(offer), this.values);

        // Filter out value combinations that yield best rate for opponent, and
        // worst for us
        return acc + (opponentValue >= 0.6 ? 1 : 0) +
            (selfValue <= 0.7 ? 1 : 0);
      }, 0);
    });

    const maxValueScore = valueScores.reduce(
        (acc, curr) => Math.max(acc, curr));

    const bestValues = this.possibleValues.filter((_, i) => {
      return valueScores[i] === maxValueScore;
    });

    const offerScores = offers.map((offer) => {
      const opponent = bestValues.reduce((acc, values) => {
        return acc + this.offerValue(offer, values);
      }, 0) / bestValues.length;

      return {
        self: this.offerValue(offer, this.values),
        opponent,
      };
    }).map((pair) => {
      return Math.min(pair.opponent, pair.self);
    });

    const maxScore = offerScores.reduce(
        (acc, score) => Math.max(acc, score), 0);
    const result = offers[offerScores.indexOf(maxScore)];
    this.log('clever=' + this.offerValue(result, this.values));
    return result;
  }
};
