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

    const dates = Object.keys(player).filter(date => date !== 'all')
        .sort().reverse();

    const latest = dates[0];

    const data = player[latest];
    const sessions = data.sessions;
    const mean = data.score / sessions;
    const meanAccepted = data.score / data.agreements;
    const acceptance = data.agreements / sessions;

    entries.push({
      hash, date: latest, mean, meanAccepted, acceptance, sessions
    });
  }

  const leaderboard = entries.filter((a) => a.sessions >= 500);

  leaderboard.sort((a, b) => b.mean - a.mean);

  console.log(leaderboard.slice(0, 10));

  const requested = process.argv[2];
  if (requested) {
    let pos = null;
    let match = null;

    for (let i = 0; i < leaderboard.length; i++) {
      match = leaderboard[i];
      if (match.hash === requested) {
        pos = i;
        break;
      }
    }

    if (pos === null) {
      console.log('Not in the leaderboard yet!');
      for (const entry of entries) {
        if (entry.hash === requested) {
          console.log(entry);
          break;
        }
      }
    } else {
      console.log(pos, match);
    }
  }
}).catch((e) => {
  throw e;
});
