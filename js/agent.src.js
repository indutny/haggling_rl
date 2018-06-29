'use strict';

const weights = {/*weights*/};

function assert(exp) {
  if (!exp)
    throw new Error('Assertion failure');
}
assert.strictEqual = (a, b) => {
  if (a !== b)
    throw new Error('Assert equal failure');
};

const MAX_COUNT = 10;
const MAX_STEPS = 1000;
const ACTION_SPACE = 5;

function matmul(vec, mat) {
  assert.strictEqual(vec.length, mat.length);

  const res = new Array(mat[0].length).fill(0);
  for (let j = 0; j < res.length; j++) {
    let acc = 0;
    for (let i = 0; i < vec.length; i++) {
      acc += vec[i] * mat[i][j];
    }
    res[j] = acc;
  }

  return res;
}

function add(a, b) {
  assert.strictEqual(a.length, b.length);

  const res = a.slice();
  for (let i = 0; i < res.length; i++) {
    res[i] += b[i];
  }
  return res;
}

function mul(a, b) {
  assert.strictEqual(a.length, b.length);

  const res = a.slice();
  for (let i = 0; i < res.length; i++) {
    res[i] *= b[i];
  }
  return res;
}

function tanh(x) {
  const res = x.slice();
  for (let i = 0; i < res.length; i++) {
    res[i] = Math.tanh(res[i]);
  }
  return res;
}

function sigmoid(x) {
  const res = x.slice();
  for (let i = 0; i < res.length; i++) {
    res[i] = 1 / (1 + Math.exp(-res[i]));
  }
  return res;
}

function relu(x) {
  const res = x.slice();
  for (let i = 0; i < res.length; i++) {
    res[i] = Math.max(0, res[i]);
  }
  return res;
}

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

class LSTM {
  constructor(kernel, bias) {
    this.kernel = kernel;
    this.bias = bias;

    this.units = (this.kernel[0].length / 4) | 0;
    this.forget_bias = new Array(this.units).fill(1);
    this.activation = tanh;

    this.initial_state = {
      c: new Array(this.units).fill(0),
      h: new Array(this.units).fill(0),
    };
  }

  call(input, state=this.initial_state) {
    const x = input.concat(state.h);

    const gateInputs = add(matmul(x, this.kernel), this.bias);

    const i = gateInputs.slice(0, this.units);
    const j = gateInputs.slice(this.units, 2 * this.units);
    const f = gateInputs.slice(2 * this.units, 3 * this.units);
    const o = gateInputs.slice(3 * this.units);

    const newC = add(mul(state.c, sigmoid(add(f, this.forget_bias))),
        mul(sigmoid(i), this.activation(j)));
    const newH = mul(this.activation(newC), sigmoid(o));

    return { result: newH, state: { c: newC, h: newH } };
  }
}

class Dense {
  constructor(kernel, bias) {
    this.kernel = kernel;
    this.bias = bias;
  }

  call(input) {
    return add(matmul(input, this.kernel), this.bias);
  }
}

class Model {
  constructor(weights) {
    this.pre = new Dense(weights['haggle/preprocess/kernel:0'],
                         weights['haggle/preprocess/bias:0']);
    this.lstm = new LSTM(weights['haggle/lstm/kernel:0'],
      weights['haggle/lstm/bias:0']);
    this.action = new Dense(weights['haggle/action/kernel:0'],
                            weights['haggle/action/bias:0']);
  }

  call(input, state) {
    const available = input.slice(0, ACTION_SPACE);
    input = input.slice(ACTION_SPACE);

    const pre = relu(this.pre.call(input));
    let { result: x, state: newState } = this.lstm.call(pre, state);
    x = this.action.call(x);

    // Mask
    x = mul(x, available);
    const probs = softmax(x);

    let roll = Math.random();
    let action = 0;
    for (;;) {
      roll -= probs[action];
      if (roll <= 0) {
        break;
      }
      action++;
    }

    return { probs, action, state: newState };
  }
}

class Environment {
  constructor(values, counts, types = 3) {
    this.types = types;

    this.position = 0;
    this.offer = new Array(MAX_COUNT).fill(0);
    this.values = new Array(MAX_COUNT).fill(0);
    this.counts = new Array(MAX_COUNT).fill(0);

    assert(values.length <= this.values.length);
    for (let i = 0; i < values.length; i++)
      this.values[i] = values[i];

    assert(counts.length <= this.counts.length);
    for (let i = 0; i < counts.length; i++)
      this.counts[i] = counts[i];
  }

  buildObservation() {
    const available = [ 1, 0, 0, 0, 0 ];

    // Cell
    const pos = this.position;
    const maxValue = this.counts[pos];
    const currentValue = this.offer[pos];
    if (currentValue !== maxValue) {
      available[1] = 1;
    }
    if (currentValue !== 0) {
      available[2] = 1;
    }

    // Movement
    if (pos !== 0) {
      available[3] = 1;
    }
    if (pos !== this.types) {
      available[4] = 1;
    }
    return [].concat(available, this.position, this.offer, this.values,
        this.counts);
  }

  setOffer(offer) {
    assert(offer.length <= this.offer.length);
    for (let i = 0; i < offer.length; i++)
      this.offer[i] = offer[i];
  }

  getOffer() {
    return this.offer.slice(0, this.types);
  }

  step(action) {
    if (action === 0) {
      return this.getOffer();
    } else if (action === 1 || action === 2) {
      this.makeChange(action === 1 ? 1 : -1);
      return false;
    } else if (action === 3 || action === 4) {
      this.move(action === 3 ? -1 : 1);
      return false;
    }
    assert(false, 'Unexpected action');
  }

  makeChange(delta) {
    const next = this.offer[this.position] + delta;
    const max = this.counts[this.position];

    this.offer[this.position] = Math.min(Math.max(next, 0), max);
  }

  move(delta) {
    const next = this.position + delta;
    this.position = Math.min(Math.max(next, 0), this.types - 1);
  }
}

const m = new Model(weights);

module.exports = class Agent {
  constructor(me, counts, values, max_rounds, log) {
    this.env = new Environment(values, counts);
    this.log = log;
    this.state = undefined;
  }

  offer(o) {
    try {
      return this._offer(o);
    } catch (e) {
      this.log(e.stack);
      throw e;
    }
  }

  _offer(o) {
    if (o) {
      this.env.setOffer(o);
    }

    let offer = undefined;
    for (let i = 0; i < MAX_STEPS; i++) {
      const { action, probs, newState } = m.call(this.env.buildObservation(),
        this.state);
      this.state = newState;

      this.log('Probabilities: ' + probs.map(p => p.toFixed(2)).join(', '));
      offer = this.env.step(action);
      if (offer) {
        break;
      }
    }

    // Timed out
    if (!offer) {
      this.log('Timed out');
      return undefined;
    }

    // First offer
    if (!o) {
      return offer;
    }

    // Success
    let accept = true;
    for (let i = 0; i < offer.length; i++) {
      if (o[i] !== offer[i]) {
        accept = false;
      }
    }

    return accept ? undefined : offer;
  }
};
