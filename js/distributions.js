'use strict';

const MAX_LEN = 1000;

function fillA(count) {
  const list = [];
  for (let i = 0; i < count; i++) {
    if (list.length < MAX_LEN) {
      list.push(i);
    } else {
      const rand = (Math.random() * list.length) | 0;
      list[rand] = i;
    }
  }
  return list;
}

function fillB(count) {
  let list = [];
  let delta = 1;
  for (let i = 0; i < count; i += delta) {
    list.push(i);
    if (list.length < 2 * MAX_LEN) {
      continue;
    }

    const half = [];
    for (let i = 0; i < list.length; i += 2) {
      half.push(list[i]);
    }
    list = half;
    delta *= 2;
  }
  return list;
}

function mean(list) {
  let res = 0;
  for (const elem of list) {
    res += elem;
  }
  return res / list.length;
}

console.log('A: %d', mean(fillA(8000)));
console.log('B: %d', mean(fillB(8000)));
