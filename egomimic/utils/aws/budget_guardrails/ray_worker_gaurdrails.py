import argparse
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import boto3

STATE_PATH = Path("/tmp/ray_worker_guardrails_state.json")


def _imdsv2_token():
    req = Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
    )
    with urlopen(req, timeout=2) as resp:
        return resp.read().decode("utf-8")


def _imdsv2_get(path):
    token = _imdsv2_token()
    req = Request(
        f"http://169.254.169.254/latest/{path}",
        headers={"X-aws-ec2-metadata-token": token},
    )
    with urlopen(req, timeout=2) as resp:
        return resp.read().decode("utf-8")


def _get_identity_doc():
    doc = _imdsv2_get("dynamic/instance-identity/document")
    return json.loads(doc)


def _get_head_cluster_name(ec2, instance_id):
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    for reservation in resp.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            for tag in instance.get("Tags", []):
                if tag.get("Key") == "ray-cluster-name":
                    return tag.get("Value")
    return None


def _run_ray_list_nodes():
    commands = [
        ["ray", "list", "nodes", "--format", "json"],
        ["ray", "state", "list", "nodes", "--format", "json"],
    ]
    last_error = None
    for cmd in commands:
        try:
            output = subprocess.check_output(cmd, text=True)
            nodes = _parse_nodes_output(output)
            if nodes:
                return nodes
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
    raise RuntimeError(f"Unable to list Ray nodes: {last_error}")


def _normalize_nodes(parsed):
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        normalized = []
        for item in parsed:
            normalized.extend(_normalize_nodes(item))
        return normalized
    return []


def _parse_nodes_output(output):
    output = output.strip()
    if not output:
        return []
    try:
        return _normalize_nodes(json.loads(output))
    except json.JSONDecodeError:
        nodes = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            nodes.extend(_normalize_nodes(json.loads(line)))
        return nodes


def _fallback_ray_nodes():
    import ray

    ray.init(address="auto", namespace="ray_worker_guardrails")
    nodes = ray.nodes()
    normalized = []
    for node in nodes:
        normalized.append(
            {
                "node_id": node.get("NodeID"),
                "node_ip": node.get("NodeManagerAddress"),
                "is_head_node": socket.gethostname() == node.get("NodeManagerHostname"),
                "state": "ALIVE" if node.get("Alive") else "DEAD",
            }
        )
    return normalized


def _load_nodes():
    try:
        return _run_ray_list_nodes()
    except RuntimeError:
        return _fallback_ray_nodes()


def _extract_node_ip(node):
    return (
        node.get("node_ip")
        or node.get("node_ip_address")
        or node.get("node_manager_address")
        or node.get("NodeManagerAddress")
        or node.get("node_ip_address")
    )


def _is_head(node, local_ip):
    if "is_head_node" in node:
        return bool(node["is_head_node"])
    node_ip = _extract_node_ip(node)
    return node_ip == local_ip


def _resource_idle(node, eps=1e-6):
    total = node.get("resources_total") or node.get("resourcesTotal")
    available = node.get("resources_available") or node.get("resourcesAvailable")
    if not isinstance(total, dict) or not isinstance(available, dict):
        return None

    total_cpu = float(total.get("CPU", 0.0))
    avail_cpu = float(available.get("CPU", 0.0))
    total_gpu = float(total.get("GPU", 0.0))
    avail_gpu = float(available.get("GPU", 0.0))
    return avail_cpu + eps >= total_cpu and avail_gpu + eps >= total_gpu


def _load_state():
    if not STATE_PATH.exists():
        return {}
    with STATE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_state(state):
    with STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def _describe_instance_by_ip(ec2, cluster_name, ip_address):
    filters = [
        {"Name": "private-ip-address", "Values": [ip_address]},
        {"Name": "instance-state-name", "Values": ["running"]},
    ]
    if cluster_name:
        filters.append({"Name": "tag:ray-cluster-name", "Values": [cluster_name]})
    resp = ec2.describe_instances(Filters=filters)
    for reservation in resp.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            return instance
    return None


def _uptime_seconds(instance, now):
    launch_time = instance.get("LaunchTime")
    if not launch_time:
        return None
    if launch_time.tzinfo is None:
        launch_time = launch_time.replace(tzinfo=timezone.utc)
    return (now - launch_time).total_seconds()


def main():
    parser = argparse.ArgumentParser(
        description="Terminate Ray worker EC2 instances that are idle or too old."
    )
    parser.add_argument("--max-uptime-hours", type=float, default=2.0)
    parser.add_argument("--max-idle-minutes", type=float, default=15.0)
    parser.add_argument("--cluster-name", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    identity = _get_identity_doc()
    region = args.region or identity["region"]
    local_ip = identity["privateIp"]

    ec2 = boto3.client("ec2", region_name=region)
    cluster_name = args.cluster_name or _get_head_cluster_name(
        ec2, identity["instanceId"]
    )

    nodes = _load_nodes()
    now = datetime.now(timezone.utc)
    state = _load_state()
    updated_state = {}

    max_uptime = args.max_uptime_hours * 3600.0
    max_idle = args.max_idle_minutes * 60.0
    to_terminate = []

    for node in nodes:
        node_state = (node.get("state") or node.get("State") or "").upper()
        if node_state and node_state not in {"ALIVE", "RUNNING"}:
            continue

        node_ip = _extract_node_ip(node)
        if not node_ip:
            continue
        if _is_head(node, local_ip):
            continue

        instance = _describe_instance_by_ip(ec2, cluster_name, node_ip)
        if not instance:
            continue

        instance_id = instance["InstanceId"]
        uptime = _uptime_seconds(instance, now)
        if uptime is not None and uptime > max_uptime:
            to_terminate.append(instance_id)
            continue

        idle_now = _resource_idle(node)
        last_active = state.get(instance_id, now.timestamp())
        if idle_now is True:
            idle_for = now.timestamp() - last_active
            if idle_for > max_idle:
                to_terminate.append(instance_id)
                continue
        elif idle_now is False:
            last_active = now.timestamp()

        updated_state[instance_id] = last_active

    if to_terminate:
        unique_ids = sorted(set(to_terminate))
        if args.dry_run:
            print(f"[dry-run] Would terminate: {', '.join(unique_ids)}")
        else:
            ec2.terminate_instances(InstanceIds=unique_ids)
            print(f"Terminated: {', '.join(unique_ids)}")
    else:
        print("No workers matched termination criteria.")

    _save_state(updated_state)


if __name__ == "__main__":
    main()
