# 🛰️ Ray Cluster: `aria-cluster.yaml`

This cluster definition launches the **Aria Ray cluster** on AWS (region `us-east-2`) with:

- **Head node:** `t3.xlarge` (4 vCPUs, 16 GiB RAM)
- **Worker nodes:** `c5.18xlarge` (72 vCPUs, 144 GiB RAM)
- Mounts S3 buckets:
  - `/mnt/raw` → `s3://rldb/raw_v2/aria`
  - `/mnt/processed` → `s3://rldb/processed_v2/aria`
- Auto-updates and runs the daily conversion cron job.

---

## ⚙️ Before launching

Update the SSH key path in the YAML if needed:

```yaml
auth:
  ssh_private_key: ~/.ssh/your-key.pem  # <-- replace with your actual private key path
```

Ensure the key matches the `KeyName` specified under each node type.

---

## 🚀 Launching the cluster

From your local machine (where Ray CLI is installed and configured with AWS credentials):

```bash
ray up aria-cluster.yaml
```

**What happens**
- Spins up the **head node**.
- Auto-launches **worker nodes** when tasks require more CPUs.
- Installs Python, Ray, S3FS, SQLAlchemy, and your `EgoVerse` repo.
- Mounts `/mnt/raw` and `/mnt/processed` from S3.
- Registers a daily cron job to run `run_aria_conversion.py` at 6 PM ET.

Monitor setup:

```bash
ray monitor aria-cluster.yaml
```

or check EC2 console logs (`/var/log/cloud-init-output.log`).

---

## 💻 Connecting to the head node

Once the head node is up:

```bash
ray attach aria-cluster.yaml
```

This SSHs into the head node (`ubuntu`) and attaches to the running Ray session.  
You can verify cluster health:

```bash
ray status
ray list nodes
ray list tasks
```

---

## 🧹 Taking down the cluster

When finished:

```bash
ray down aria-cluster.yaml
```

This terminates **all head + worker EC2 instances** and cleans up security groups, volumes, etc.  
Use `--yes` to skip confirmation.

---