# Ingest on AWS

One small always-on box runs the loop (`egomimic.ingest.run`: a watcher pass, then the
committer, every 5 minutes). Conversions run as AWS Batch jobs on Spot GPU instances that
scale to zero. Lance and the staging prefix live on R2 (`rldb`); vendor uploads and the
pipeline's records live in the private `ingest` and `ingest-<vendor>` buckets.

```
vendor ──S3 tools / dashboard──▶ ingest-<vendor>/<delivery>/        (_done.json or _complete.json)
box: watch pass ──submit──▶ Batch job: convert_job ──▶ ingest/pipeline/staged/<vendor>/<delivery>/<digest>/
staff: python -m egomimic.ingest.review approve ...   (or the dashboard)
box: watch pass ──server-side copy──▶ rldb/raw_v2/<vendor>/<delivery>/<digest>/
                                  └──▶ rldb/lance_staging/<source>/<vendor recording id>/
box: lance_commit ──▶ rldb/lance/  (+ changes row, tag <source>-<UTC>)
```

## Cost and speed (measured 2026-10-21)

One 3-minute Aria recording (5,421 frames) converted in 735 s on an A40 with 8 cores
(MANO 553 s of it): about 4 machine-hours per footage-hour, and 2.3 GB of output per
footage-hour. MANO on 8 CPU cores alone ran about 14x realtime, so jobs take a GPU.
At ~$0.35-0.5/h for a Spot g6/g5.2xlarge that is roughly $1.4-2k of compute per
1,000 footage-hours, plus AWS's ~$0.09/GB for sending output to R2 (~$200 at 2.3 TB).
`--mano-iters` (2400) is the lever if that is too much.

## One-time setup

1. **Image**: build and push to a private ECR repository. The MANO model files are
   license-gated and come in as a build secret, never in git:
   ```
   tar -cf /tmp/mano.tar -C external_ckpts/mano MANO_LEFT.pkl MANO_RIGHT.pkl
   docker build -f egomimic/ingest/aws/Dockerfile --build-arg REVISION=$(git rev-parse HEAD) \
       --secret id=mano,src=/tmp/mano.tar -t $ECR/egoverse-ingest:$(git rev-parse --short HEAD) .
   docker push $ECR/egoverse-ingest:$(git rev-parse --short HEAD)
   ```
2. **R2 credentials** for the box and jobs in Secrets Manager (`R2_ENDPOINT_URL`,
   `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`), with write access to `ingest`, every
   `ingest-*` bucket and `rldb`.
3. **Batch**: the Spot vCPU quota for G instances (`L-3819A6DF`, "All G and VT Spot
   Instance Requests") must cover `--max-vcpus`. In the lab account (us-east-2) it was 0
   on 2026-10-21, as was the On-Demand one (`L-DB2E81BA`), so request it first:
   `aws service-quotas request-service-quota-increase --service-code ec2
   --quota-code L-3819A6DF --desired-value 448 --region us-east-2`. Then
   ```
   python -m egomimic.ingest.aws.setup_batch --image $ECR/egoverse-ingest:<rev> \
       --subnets ... --security-group ... --instance-role ecsInstanceRole \
       --job-role <role that can read the secret> --env-secret <secret arn> --dry-run
   ```
   and again without `--dry-run`.
4. **Box**: a `t3.small` (x86, so it runs the same image) with an instance role allowing
   `batch:SubmitJob` / `batch:DescribeJobs` on the queue and reading the secret:
   ```
   docker run -d --restart=always --name ingest -v /var/lib/ingest:/work \
       --env-file /etc/ingest/r2.env $ECR/egoverse-ingest:<rev> \
       python -m egomimic.ingest.run --lance s3://rldb/lance --work /work --launcher batch
   ```
   `touch /var/lib/ingest/STOP` ends the loop after the current round.

## Per vendor

An admin, from their own machine, with a Cloudflare token that can create buckets and
API tokens:
```
CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... \
    python -m egomimic.ingest.onboard scale --dashboard-origin https://<dashboard host>
```
It prints the vendor's bucket, endpoint and keys once. The vendor uploads each delivery
to `s3://ingest-scale/<delivery id>/` and writes `_done.json` last; to correct one, delete
`_done.json` first.

## Smoke test

Run everything against an isolated namespace before the first real delivery:
`--root _smoke/<date>/` on the box and in the job arguments puts every ingest key under
that prefix, and `--lance s3://rldb/scratch/<you>/lance --lance-staging
s3://rldb/scratch/<you>/lance_staging --raw-root s3://rldb/scratch/<you>/raw` keeps Lance
untouched. Upload one delivery, run `python -m egomimic.ingest.run --once`, approve it with
the review CLI, run `--once` twice more, and check the recording in the scratch tables.
