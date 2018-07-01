'use strict';

const fetch = require('node-fetch');

let bestMean = 0;
let best = null;

function process(hash, date, entry) {
  if (entry.sessions < 1000) {
    return;
  }

  const mean = entry.score / entry.sessions;
  if (mean > bestMean) {
    bestMean = mean;
    best = { hash, date, entry };
  }
}

fetch('https://hola.org/challenges/haggling/scores/standard').then((res) => {
  return res.json();
}).then((json) => {
  let entries = [];
  for (const hash of Object.keys(json)) {
    const player = json[hash];

    for (const date of Object.keys(player)) {
      if (date === 'all') {
        continue;
      }

      const data = player[date];
      const sessions = data.sessions;
      const mean = data.score / sessions;
      const meanAccepted = data.score / data.agreements;
      const acceptance = data.agreements / sessions;

      entries.push({ hash, date, mean, meanAccepted, acceptance, sessions });
    }
  }

  entries = entries.filter((a) => a.sessions >= 500);

  entries.sort((a, b) => b.mean - a.mean);

  console.log(entries.slice(0, 10));
}).catch((e) => {
  throw e;
});
