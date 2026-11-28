# Original non-look-at Cart fix — FT60k

Run: egoa180-cart-fix1124-ft60k-MSCstrong-s42. Exact60k, original EgoVerse180k initialization, strong MSC, seed42. Own corrected Cart30/10 target data; NOT sharedfix1124 look-at. Port8282, TGZ10HD2, virtual camera ON, standard RTC8/10/13.

Download on robotpc:
```bash
mkdir -p /home/aloha/RB_Y1_workspace/policies/egoa180-cart-fix-FT60k-robotpc-20261124
rsync -rt --links --info=progress2 \
  -e "ssh -o IdentitiesOnly=yes" \
  cxin34@sky1.cc.gatech.edu:/coc/flash7/cxin34/egoa180-cart-fix-FT60k-robotpc-20261124/ \
  /home/aloha/RB_Y1_workspace/policies/egoa180-cart-fix-FT60k-robotpc-20261124/
cd /home/aloha/RB_Y1_workspace/policies/egoa180-cart-fix-FT60k-robotpc-20261124
sha256sum -c SHA256SUMS
```

Terminal1 from that directory:
```bash
bash scripts/serve.sh
```

Terminal2:
```bash
cd /home/aloha/RB_Y1_workspace/policies/egoa180-cart-fix-FT60k-robotpc-20261124
bash scripts/client.sh inspect-client && bash scripts/check.sh && bash scripts/client.sh dry-run
```

Operator-controlled physical rollout after local checks:
```bash
bash scripts/client.sh run
```

Stop any prior robot-control client before switching. Same accepted camera/reset/gaze as this run's original10k package; no look-at camera substitution. Matching code/config/source-frozen normalizer included. Packaging checks are CPU offline; robotpc CUDA/installed-client/hardware tests remain local. No camera or robot motion was started by packaging.
