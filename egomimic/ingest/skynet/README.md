# The ingestion watcher on SkyNet

`watch.sbatch` runs one pass of `python -m egomimic.ingest.watch` (stage new deliveries,
commit approved ones) as a CPU job on the `overcap` partition, then the next pass
`INGEST_INTERVAL_MIN` (default 15) minutes after it ends. The next job is queued before
each pass runs, so a crash, timeout or preemption does not end the chain. A pass holds
locks on the work folder and the Lance root; run **one** watcher per Lance root.

Lab rules apply: nothing runs on `sky1` itself (only `sbatch`, `squeue`, `scancel`, file
copies); use a Slurm session for anything else, including building the venv.

## Layout (`INGEST_ROOT`)

```
$INGEST_ROOT/
  code/<revision>/   deployed code (deploy.sh), code/current -> the active revision
  venv/              Python environment for the watcher
  models/mano/       MANO_LEFT.pkl, MANO_RIGHT.pkl (license-gated, not in git)
  egoverse_env       R2 credentials (write access to the bucket), chmod 600
  lance/             the Lance root the watcher commits to
  work/              downloads and staging scratch (cleared per delivery)
  logs/<job>.out     one log per pass
  STOP               present = the chain stops at its next job
```

`/coc/flash7/scratch` is purged after 30 days unused. Staged results live in the bucket,
so only `lance/` matters long term: ask the SkyNet admins for `/coc/flash7/<user>` and
point `INGEST_ROOT` there before real use.

## One-time setup

1. **Code**, from a laptop with the repo: `egomimic/ingest/skynet/deploy.sh sky1 $INGEST_ROOT`.
2. **Venv**, inside a Slurm session (`srun -p overcap -c 8 --mem 32G --pty bash`), with
   [uv](https://docs.astral.sh/uv/) installed in your home (it is not on SkyNet); the full
   lock includes torch, so this takes a few GB under `INGEST_ROOT`:
   ```
   cd $INGEST_ROOT/code/current
   uv venv $INGEST_ROOT/venv --python 3.11
   VIRTUAL_ENV=$INGEST_ROOT/venv uv sync --active --locked
   uv pip install --python $INGEST_ROOT/venv/bin/python pylance "torchcodec==0.4.*"
   ```
   **MANO** (on by default: Aria-layout hands are fitted to MANO): install the project's
   `mano` package into the venv once, and put the license-gated `MANO_LEFT.pkl` /
   `MANO_RIGHT.pkl` in `$INGEST_ROOT/models/mano/` (or set `INGEST_MANO_MODEL_DIR`). They
   stay out of the code, so deployments stay small and immutable. A pass checks both
   before it starts; if either is missing the job stops the chain with one error rather
   than failing every delivery. `INGEST_MANO=0` turns the fit off (hands stored in the
   Aria layout, as `aria_to_lance --no-mano`). Each stage pins the setting and a sha256
   of the model files.
3. **Credentials**: an R2 token with read and write access to the bucket in
   `$INGEST_ROOT/egoverse_env` (`R2_ENDPOINT_URL=...`, `R2_ACCESS_KEY_ID=...`,
   `R2_SECRET_ACCESS_KEY=...`), then `chmod 600`.
4. **ffmpeg** with libsvtav1: Ryan's build is used by default
   (`INGEST_FFMPEG_DIR=/coc/flash7/rco3/ffmpeg-7.1-shared`).

Optional: `INGEST_BUCKET` (default `rldb`), `INGEST_PREFIX` (a key prefix, e.g.
`_test/run1/` for a trial), `INGEST_INTERVAL_MIN`, `INGEST_MANO` / `INGEST_MANO_MODEL_DIR`, `INGEST_LANCE_ROOT` /
`INGEST_WORK` (default `lance/`, `work/` under the root), `INGEST_ONCE=1` (one pass, no
chain).

## Smoke test before the chain

Before starting the recurring chain on a new setup or a new revision, run one real
delivery through an isolated bucket prefix **and** an isolated Lance root, one pass at a
time. `INGEST_PREFIX` only isolates the bucket: without `INGEST_LANCE_ROOT` the approved
test recording would be committed to the real Lance root.

```
R=/coc/flash7/scratch/mcap_pipeline
SMOKE="INGEST_ROOT=$R,INGEST_PREFIX=_test/smoke-$(date +%Y%m%d)/,INGEST_LANCE_ROOT=$R/smoke/lance,INGEST_WORK=$R/smoke/work,INGEST_ONCE=1"
ssh sky1 "bash -lc 'sbatch --output=$R/logs/smoke-%j.out --export=ALL,$SMOKE \
    $R/code/current/egomimic/ingest/skynet/watch.sbatch'"
```

1. Upload one real Aria MCAP as vendor `smoke` (vendor names are lowercase slugs, no
   leading `_`) under that prefix: `R2Store(root="_test/smoke-<date>/")
   .upload_delivery(Ref("smoke", "<id>"), folder, {...})`.
2. One pass (the command above; `INGEST_ONCE=1` queues no next job): the status must
   reach `ready`.
3. Approve it: `store.add_decision(ref, "approve", who, stage=status["stage"])`.
4. A second pass (same command): the status must reach `committed`.
5. Check the stage's pin (converter revision = the deployed `REVISION`, the MANO model
   digest) and `$R/smoke/lance` (one recording, its steps, readable videos); the real
   `$R/lance` must be untouched.
6. Delete the `_test/smoke-<date>/` prefix in R2 and `$R/smoke/`.

## Start, watch, stop

From a laptop (`R` is the `INGEST_ROOT` on SkyNet; Slurm tools need a login shell):

```
R=/coc/flash7/scratch/mcap_pipeline

# start (once; the chain keeps itself going)
ssh sky1 "bash -lc 'rm -f $R/STOP; sbatch --output=$R/logs/%j.out \
    --export=ALL,INGEST_ROOT=$R $R/code/current/egomimic/ingest/skynet/watch.sbatch'"

# what is queued / running, and the latest log
ssh sky1 "bash -lc 'squeue -u \$USER -n egoverse-ingest'"
ssh sky1 "tail -n 30 \$(ls -t $R/logs/*.out | head -1)"

# stop: no new passes; the one running finishes
ssh sky1 "touch $R/STOP"
# stop now
ssh sky1 "bash -lc 'touch $R/STOP; scancel -u \$USER -n egoverse-ingest'"
```

A pass prints a JSON summary (staged / rejected / failed / committed / declined / stale);
per-delivery detail is in each delivery's status in the bucket
(`pipeline/status/<vendor>/<delivery>.json`), which the dashboard shows.
