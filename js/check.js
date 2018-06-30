'use strict';

const Agent = require('./agent');

const log = (msg) => {
  console.log(msg);
};

const a = new Agent(null, [ 2, 2, 1 ], [ 1, 3, 1 ], 5, log);

console.log(a.offer());
console.log(a.offer());
