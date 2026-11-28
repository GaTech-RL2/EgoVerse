"""Check a local deployment package against its stored CPU action fixtures.

Loads only the checkpoint tensor state with weights_only=True. No server,
network, robot connection, package-path imports, or remote downloads.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from campaign.cart_sim import CampaignSimPolicy, PROPRIO
from campaign.data import sha
from campaign.policy import CampaignPolicy


def check(package):
    package = Path(package)
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    normalizer = json.loads((package/'normalizer.json').read_text())
    checkpoint = torch.load(package/'checkpoint.pt', map_location='cpu', weights_only=True)
    assert checkpoint['identity']['normalizer_sha256'] == sha(package/'normalizer.json')
    assert checkpoint['identity']['config_sha256'] == sha(package/'configs/training.json')
    robot = json.loads((root/'assets/robot.json').read_text())
    robot['urdf'] = str(root/'assets/robot.urdf')
    model = CampaignPolicy(root/'configs/hd2-reference.yaml', 'B', normalizer, robot)
    model.load_state_dict(checkpoint['model'], strict=True)
    key_count = len(checkpoint['model'])
    step = checkpoint['step']
    del checkpoint
    model.policy.heads['rby1'].num_inference_steps = 16
    policy = CampaignSimPolicy(model.eval(), normalizer, 'B')
    cases = []
    with np.load(package/'testdata/reference.npz', allow_pickle=False) as fixture:
        for i in range(3):
            obs = {k:fixture[f'{i}_{k}'].copy()
                   for k in ['front_pcd_1','front_pcd_2','front_pcd_3',*PROPRIO]}
            # Exact deployed constant-proprio behavior, before normalization.
            for raw,key in PROPRIO.items():
                lo = np.asarray(normalizer[key]['quantile_1'])
                hi = np.asarray(normalizer[key]['quantile_99'])
                obs[raw][hi-lo < 1e-6] = lo[hi-lo < 1e-6]
            obs['_simulation_seed'] = 123+i
            result = policy.infer(obs)['actions']
            delta = np.abs(result-fixture[f'{i}_plain'])
            assert result.shape == (64,49) and np.isfinite(result).all()
            assert delta.mean() < .001 and delta.max() < .01, (i,delta.mean(),delta.max())
            obs.update(rtc_prev_actions=fixture[f'{i}_plain'], rtc_freeze_steps=8,
                       rtc_soft_horizon=13, rtc_soft_power=1., _simulation_seed=223+i)
            rtc = policy.infer(obs)['actions']
            rtc_delta = np.abs(rtc-fixture[f'{i}_rtc'])
            assert np.isfinite(rtc).all()
            assert rtc_delta.mean() < .001 and rtc_delta.max() < .01
            np.testing.assert_allclose(rtc[:8],fixture[f'{i}_plain'][:8],rtol=1e-5,atol=1e-6)
            cases.append(dict(frame=i,plain_mean=float(delta.mean()),plain_max=float(delta.max()),
                              rtc_mean=float(rtc_delta.mean()),rtc_max=float(rtc_delta.max())))
    return dict(passed=True,checkpoint_step=step,strict_state_keys=key_count,
                checkpoint_sha256=sha(package/'checkpoint.pt'),
                normalizer_sha256=sha(package/'normalizer.json'),
                source='Nov19 Basket B deployment; not latest Nov25 checkpoint',
                device='cpu',network_used=False,hardware_used=False,cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path)
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    result = check(args.package)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
