'use strict';

const fetch = require('node-fetch');

let bestMean = 0;
let best = null;

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

  const requested = process.argv[2];
  if (requested) {
    let pos = null;
    let entry = null;

    for (let i = 0; i < entries.length; i++) {
      entry = entries[i];
      if (entry.hash === requested) {
        pos = i;
        break;
      }
    }

    console.log(pos, entry);
  }
}).catch((e) => {
  throw e;
});
