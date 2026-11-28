# Shared fix1124 Cart look-at / Basket55 FT10k

Exact step10000 of the two requested seed42, strong-MSC runs. The60k in the run name is the training target, not the exported step.

| Task | Package | Port | Native preset |
|---|---|---|---|
| Cart look-at | egoa180-cart-la-sharedfix1124-FT10k-robotpc-20261124 | 8280 | TGZ10HD2 with package-specific look-at VV_MODEL |
| Basket55 | egoa180-basket55-sharedfix1124-FT10k-robotpc-20261124 | 8281 | TGZ10HD2Z5 |

Download on robotpc:
```bash
for pkg in egoa180-cart-la-sharedfix1124-FT10k-robotpc-20261124 egoa180-basket55-sharedfix1124-FT10k-robotpc-20261124; do
  mkdir -p "/home/aloha/RB_Y1_workspace/policies/$pkg" || break
  rsync -rt --links --info=progress2 -e "ssh -o IdentitiesOnly=yes" \
    "cxin34@sky1.cc.gatech.edu:/coc/flash7/cxin34/$pkg/" \
    "/home/aloha/RB_Y1_workspace/policies/$pkg/" || break
  (cd "/home/aloha/RB_Y1_workspace/policies/$pkg" && sha256sum -c SHA256SUMS) || break
done
```

Set the same chosen package in both terminals:
```bash
PKG=egoa180-cart-la-sharedfix1124-FT10k-robotpc-20261124
# For Basket instead:
# PKG=egoa180-basket55-sharedfix1124-FT10k-robotpc-20261124
cd "/home/aloha/RB_Y1_workspace/policies/$PKG"
```

Terminal1:
```bash
bash scripts/serve.sh
```

Terminal2, local client/CUDA verification:
```bash
bash scripts/client.sh inspect-client && bash scripts/check.sh && bash scripts/client.sh dry-run
```

Operator-controlled hardware execution after local checks:
```bash
bash scripts/client.sh run
```

These wrappers use RTC8/10/13,16 flow steps, virtual camera ON, task-specific train-only reset/camera. They use the standard native launcher, not the separate background-observation12/13/21 launcher. Stop the prior native client before switching policies. No runtime Cart head override; do not substitute older Cart camera assets. Only offline CPU/native-asset verification was performed by packaging; robotpc CUDA/client and hardware are not certified by those checks.
