# PytracIK Installation for Kinematics

This guide explains how to install **PytracIK** (TRAC-IK Python bindings) entirely in user space — no `sudo` required.

## 1. Clone the repository (without submodules)
To avoid nested Git submodules, clone the repo into a temporary directory, strip its `.git` folder, and move it to your working directory:
```bash
git clone https://github.com/chenhaox/pytracik.git pytracik-temp
rm -rf pytracik-temp/.git
mv pytracik-temp ./pytracik
cd pytracik
uv pip install -r requirements.txt / pip install -r requirements.txt
```

## 2. Install dependencies inside your conda environment
Make sure you’re in your desired environment (for example, `emimic`):
```bash
conda activate emimic
conda install -y --override-channels -c conda-forge --no-channel-priority boost-cpp eigen orocos-kdl nlopt
```

## 3. Build and install PytracIK
```bash
python setup_linux.py install
```

## 4. Verify installation
```bash
python - <<'PY'
from trac_ik import TracIK
print("PytracIK successfully installed and importable.")
PY
```