"""Create the AWS Batch pieces the conversion jobs run on: a Spot compute environment of
GPU instances that scales to zero, a job queue, and a job definition for the ingest image.

    python -m egomimic.ingest.aws.setup_batch --image <account>.dkr.ecr.<region>.amazonaws.com/egoverse-ingest:<rev> \\
        --subnets subnet-a subnet-b --security-group sg-x --instance-role ecsInstanceRole \\
        --job-role egoverse-convert-job --env-secret arn:aws:secretsmanager:...:r2 --dry-run

MANO runs on the GPU: one 3-minute Aria recording took 735 s on an A40 with 8 cores, about
4x realtime, against roughly 14x on 8 CPU cores alone. Each job takes one GPU and 8 vCPUs,
so --max-vcpus 448 runs up to 56 conversions at once. Interrupted Spot jobs are retried;
the watcher fails a delivery whose job ends without an outcome.
"""

from __future__ import annotations

import argparse
import json
import sys

NAME = "egoverse-convert"
INSTANCE_TYPES = ["g6.2xlarge", "g5.2xlarge"]  # 1 GPU, 8 vCPU each


def requests(args) -> list[tuple[str, dict]]:
    """The Batch API calls, in order, as (method, kwargs)."""
    env = {
        "computeEnvironmentName": args.name,
        "type": "MANAGED",
        "state": "ENABLED",
        "computeResources": {
            "type": "SPOT",
            "allocationStrategy": "SPOT_PRICE_CAPACITY_OPTIMIZED",
            "minvCpus": 0,
            "maxvCpus": args.max_vcpus,
            "instanceTypes": args.instance_types,
            "subnets": args.subnets,
            "securityGroupIds": [args.security_group],
            "instanceRole": args.instance_role,
            "ec2Configuration": [{"imageType": "ECS_AL2_NVIDIA"}],
            "tags": {"project": "egoverse-ingest"},
        },
    }
    queue = {
        "jobQueueName": args.name,
        "state": "ENABLED",
        "priority": 1,
        "computeEnvironmentOrder": [{"order": 1, "computeEnvironment": args.name}],
    }
    container = {
        "image": args.image,
        "command": ["python", "-m", "egomimic.ingest.convert_job", "--help"],
        "jobRoleArn": args.job_role,
        "resourceRequirements": [
            {"type": "VCPU", "value": str(args.vcpus)},
            {"type": "MEMORY", "value": str(args.memory_mib)},
            {"type": "GPU", "value": "1"},
        ],
        "environment": [{"name": "PYTHONUNBUFFERED", "value": "1"}],
        "secrets": [
            {"name": name, "valueFrom": f"{args.env_secret}:{name}::"}
            for name in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
        ],
        "volumes": [{"name": "scratch", "host": {"sourcePath": "/scratch"}}],
        "mountPoints": [{"sourceVolume": "scratch", "containerPath": "/scratch"}],
    }
    definition = {
        "jobDefinitionName": args.name,
        "type": "container",
        "containerProperties": container,
        "retryStrategy": {
            "attempts": 3,
            "evaluateOnExit": [
                {"onStatusReason": "Host EC2*", "action": "RETRY"},  # Spot reclaimed
                {"onReason": "*", "action": "EXIT"},
            ],
        },
        "timeout": {"attemptDurationSeconds": args.timeout_s},
    }
    return [
        ("create_compute_environment", env),
        ("create_job_queue", queue),
        ("register_job_definition", definition),
    ]


def main(argv: list[str] | None = None, client=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--image", required=True)
    ap.add_argument("--subnets", nargs="+", required=True)
    ap.add_argument("--security-group", required=True)
    ap.add_argument("--instance-role", required=True)
    ap.add_argument("--job-role", required=True)
    ap.add_argument(
        "--env-secret", required=True, help="Secrets Manager ARN with the R2_* keys"
    )
    ap.add_argument("--instance-types", nargs="+", default=INSTANCE_TYPES)
    ap.add_argument("--max-vcpus", type=int, default=448)
    ap.add_argument("--vcpus", type=int, default=8)
    ap.add_argument("--memory-mib", type=int, default=28_000)
    ap.add_argument("--timeout-s", type=int, default=6 * 3600)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    calls = requests(args)
    if args.dry_run:
        for method, kw in calls:
            print(method, json.dumps(kw, indent=1))
        return 0
    if client is None:
        import boto3

        client = boto3.client("batch")
    for method, kw in calls:
        out = getattr(client, method)(**kw)
        print(
            method,
            out.get("computeEnvironmentArn")
            or out.get("jobQueueArn")
            or out.get("jobDefinitionArn"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
