// Snapshot the available checkpoints for each fresh history run and its resumes.
import fs from 'node:fs';
import path from 'node:path';

const root = '/workspace/users/ani-cheluva/EgoVerse';
const runs = path.join(root, 'logs/fixed_budget_opscale_hist');
const out = path.join(runs, 'offline_sweep_20261010');
fs.mkdirSync(out, {recursive: true});

for (const op of [4, 8, 12, 16]) {
  const selected = new Map();
  for (const dir of fs.readdirSync(runs).sort()) {
    if (!new RegExp(`^rdt_vitb_op${op}_(fresh|resume)_\\d+$`).test(dir)) continue;
    const checkpointDir = path.join(runs, dir, 'checkpoints');
    const norm = path.join(runs, dir, 'norm_stats/norm_stats.json');
    if (!fs.existsSync(checkpointDir)) continue;
    if (!fs.existsSync(norm)) throw new Error(`Missing norm stats: ${norm}`);
    for (const name of fs.readdirSync(checkpointDir)) {
      const match = /^epoch_epoch=(\d+)\.ckpt$/.exec(name);
      if (!match) continue;
      const epoch = Number(match[1]);
      if (!(epoch <= 299 || (epoch >= 499 && (epoch - 499) % 200 === 0))) continue;
      const checkpoint = path.join(checkpointDir, name);
      if (fs.statSync(checkpoint).size === 0) continue;
      if (selected.has(epoch)) throw new Error(`Duplicate op${op} epoch ${epoch}`);
      selected.set(epoch, {checkpoint, norm});
    }
  }
  const rows = [...selected].sort((a, b) => a[0] - b[0]);
  if (!rows.length || rows[0][0] !== 49 || !rows.some(([epoch]) => epoch === 299)) {
    throw new Error(`Incomplete early checkpoint set for op${op}`);
  }
  const manifest = path.join(out, `op${op}-checkpoints.tsv`);
  const fresh = fs.readdirSync(runs).find(dir =>
    new RegExp(`^rdt_vitb_op${op}_fresh_\\d+$`).test(dir));
  const configDir = path.join(runs, fresh, '.hydra');
  if (!fs.existsSync(path.join(configDir, "config.yaml"))) throw new Error(`Missing saved run config: ${configDir}`);
  fs.writeFileSync(manifest, rows.map(([epoch, {checkpoint, norm}]) =>
    `${epoch}\t${checkpoint}\t${norm}\t${configDir}\n`).join(''));
  console.log(`op${op}: ${rows.length} checkpoints, epochs ${rows.map(([e]) => e).join(',')}`);
}
