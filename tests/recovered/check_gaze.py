"""H2-G integration/security-of-information checks; synthetic, never release readiness."""
import hashlib, json, os, time, faulthandler
faulthandler.dump_traceback_later(300, repeat=True)
print('Importing H2-G test dependencies', flush=True)
from pathlib import Path
import torch
from campaign.policy import CampaignPolicy
from campaign.gaze_policy import GazePolicy
from campaign.gaze import quantize_gaze, gaze_ce, BINS

def digest(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    print('Initializing paired models', flush=True)
    root=Path(__file__).resolve().parents[2]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    robot=json.loads((root/'assets/robot.json').read_text());robot['urdf']=str(root/'assets/robot.urdf')
    norm={'action':{'center':[0.]*49,'scale':[1.]*49}}
    torch.manual_seed(42)
    baseline=CampaignPolicy(root/'configs/hd2-reference.yaml','B',norm,robot)
    aug_names=('pose_jitter_deg','pose_jitter_m','point_noise_std_m','colour_brightness_jitter','colour_channel_jitter')
    aug={k:{name:getattr(v,name) for name in aug_names} for k,v in baseline.policy.encoders.items()}
    common={k:digest(v) for k,v in baseline.state_dict().items()};del baseline
    torch.manual_seed(42)
    m=GazePolicy(root/'configs/hd2-reference.yaml',norm,robot).to(device).eval()
    assert all(digest(m.state_dict()[k])==v for k,v in common.items())
    assert all(getattr(m.policy.encoders[k],name)==value for k,fields in aug.items() for name,value in fields.items())
    print('Paired initialization passed; checking information barriers', flush=True)
    n=2
    b={f'front_pcd_{i}':torch.rand(n,1,1,p*6,device=device) for i,p in enumerate((2048,1024,1024),1)}
    b.update({k:torch.rand(n,1,d,device=device) for k,d in [('state_robot0_joint_pos',22),('state_eef_pose_glass',9),('state_hand_left_qpos',12),('state_hand_right_qpos',12)]})
    b.update(clean_cloud=torch.rand(n,2048,6,device=device),gaze_uv=torch.tensor([[.2,.4],[.7,.8]],device=device),
             gaze_valid=torch.ones(n,dtype=torch.bool,device=device),action=torch.rand(n,64,49,device=device)*.1,
             action_valid=torch.ones(n,64,dtype=torch.bool,device=device),dino=torch.rand(n,16,768,device=device),
             dino_valid=torch.ones(n,dtype=torch.bool,device=device))
    labels=quantize_gaze(b['gaze_uv'],b['gaze_valid']);valid=b['gaze_valid']
    with torch.no_grad():
        encoded=m.encode(b);features,logits=m.run_trunk(encoded,labels,valid)
        assert all(getattr(m.policy.encoders[k],name)==value for k,fields in aug.items() for name,value in fields.items()),'Clean prefix must restore hd2 augmentation settings'
        changed_y=labels.clone();changed_y[:,1]=(changed_y[:,1]+50)%BINS
        fy,ly=m.run_trunk(encoded,changed_y,valid)
        torch.testing.assert_close(logits,ly,rtol=0,atol=0)
        assert not torch.allclose(features,fy), 'Action must use gaze y'
        changed_x=labels.clone();changed_x[:,0]=(changed_x[:,0]+70)%BINS
        _,lx=m.run_trunk(encoded,changed_x,valid)
        torch.testing.assert_close(logits[:,0],lx[:,0],rtol=0,atol=0)
        assert not torch.allclose(logits[:,1],lx[:,1]), 'y must condition on x'
        _,context_changed=m.run_trunk((encoded[0],torch.randn_like(encoded[1])*100),labels,valid)
        torch.testing.assert_close(logits,context_changed,rtol=0,atol=0)
        # Exercise the actual encoder path, not just the attention mask utility.
        corrupt={k:(torch.randn_like(v)*10 if k.startswith(('state_','front_pcd_')) else v) for k,v in b.items()}
        _,corrupt_logits=m.run_trunk(m.encode(corrupt),labels,valid)
        torch.testing.assert_close(logits,corrupt_logits,rtol=0,atol=0)
        invalid=torch.zeros_like(valid)
        f0,_=m.run_trunk(encoded,labels,invalid)
        f1,_=m.run_trunk(encoded,changed_x,invalid)
        torch.testing.assert_close(f0,f1,rtol=0,atol=0)
        a0,_=m.run_trunk(encoded,labels,valid,condition=False)
        a1,_=m.run_trunk(encoded,changed_y,valid,condition=False)
        torch.testing.assert_close(a0,a1,rtol=0,atol=0)
        predicted,pred=m.predicted_features(encoded)
        teacher,seq=m.run_trunk(encoded,pred,valid)
        torch.testing.assert_close(predicted,teacher,rtol=0,atol=0)
        assert torch.equal(seq.argmax(-1),pred), 'Shifted teacher/sequential logits disagree'
        poisoned=quantize_gaze(torch.full_like(b['gaze_uv'],float('nan')),invalid)
        assert gaze_ce(logits,poisoned,invalid)==0
        # Main eval losses ignore GT gaze and GT validity (diagnostic CE may differ).
        state=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
        loss,_=m(b)
        torch.set_rng_state(state);torch.cuda.set_rng_state_all(cuda)
        other={**b,'gaze_uv':torch.full_like(b['gaze_uv'],float('nan')),'gaze_valid':invalid}
        loss2,_=m(other);torch.testing.assert_close(loss,loss2,rtol=0,atol=0)
    # Action features backpropagate into gaze embeddings; classifier requires CE.
    encoded=m.encode(b);features,logits=m.run_trunk(encoded,labels,valid)
    features.square().mean().backward(retain_graph=True)
    assert m.gaze.x.weight.grad.abs().sum()>0 and m.gaze.y.weight.grad.abs().sum()>0
    assert m.gaze.classifier.weight.grad is None
    m.zero_grad(set_to_none=True);gaze_ce(logits,labels,valid).backward()
    assert m.gaze.classifier.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.policy.encoders['front_pcd_1'].parameters())
    m.zero_grad(set_to_none=True)
    print('Masks, sequential parity and gradient paths passed', flush=True)
    # Full deployment has no gaze labels in its API; 64 ODE steps, finite 49D actions.
    obs={k:v[:1] for k,v in b.items() if k.startswith(('front_pcd_','state_')) or k=='clean_cloud'}
    start=time.monotonic()
    result=m.predict_actions(obs)
    print('Deployment inference passed', flush=True)
    latency=time.monotonic()-start
    assert result['action'].shape==(1,64,49) and torch.isfinite(result['action']).all()
    # Requested production batch, including an all-invalid-gaze update.
    bs=int(os.getenv('CHECK_BATCH','2'))
    batch={k:v[:1].expand(bs,*v.shape[1:]).clone() for k,v in b.items()}
    m.train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
    reports=[]
    for missing in (False,True):
        if missing:
            batch['gaze_valid'].zero_();batch['gaze_uv'].fill_(float('nan'))
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):loss,parts=m(batch)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        if missing: assert m.gaze.classifier.weight.grad.abs().sum()==0 and m.gaze.x.weight.grad.abs().sum()==0
        opt.step();reports.append({k:v.item() for k,v in parts.items()})
    print(json.dumps(dict(passed=True,synthetic_only=True,paired_initialization=True,shifted_ce=True,
          every_layer_no_proprio_or_policy_leak=True,invalid_attention_mask=True,sequential_parity=True,
          action_gaze_gradient=True,ce_encoder_gradient=True,label_free_inference=True,
          batch=bs,device=device,full_precision_inference_seconds=latency,losses=reports,
          parameters=sum(p.numel() for p in m.parameters()),
          peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None),indent=2))
    faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':main()
