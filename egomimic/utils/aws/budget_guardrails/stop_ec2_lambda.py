import os

import boto3


def _build_filters():
    filters = [{"Name": "instance-state-name", "Values": ["running"]}]
    tag_key = os.environ.get("STOP_TAG_KEY")
    tag_value = os.environ.get("STOP_TAG_VALUE")
    if tag_key and tag_value:
        filters.append({"Name": f"tag:{tag_key}", "Values": [tag_value]})
    return filters


def lambda_handler(event, context):
    region = os.environ.get("REGION", "us-east-2")
    ec2 = boto3.client("ec2", region_name=region)

    paginator = ec2.get_paginator("describe_instances")
    instance_ids = []
    for page in paginator.paginate(Filters=_build_filters()):
        for reservation in page.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                instance_ids.append(instance["InstanceId"])

    if not instance_ids:
        return {"stopped": 0}

    ec2.stop_instances(InstanceIds=instance_ids)
    return {"stopped": len(instance_ids)}
