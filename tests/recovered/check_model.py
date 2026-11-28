"""Synthetic integration check only, never a data-readiness or formal-run gate."""
import json,os,time,copy
from pathlib import Path
import torch
from campaign.policy import CampaignPolicy
from campaign.objectives import flow_loss

def main():
    torch.set_num_threads(4);root=Path(__file__).resolve().parents[2]
    device='cuda' if torch.cuda.is_available() else 'cpu';bs=int(os.getenv('CHECK_BATCH','2'))
    robot=json.loads((root/'assets/robot.json').read_text());robot['urdf']=str(root/'assets/robot.urdf')
    norm=dict(action=dict(center=[0.]*49,scale=[1.]*49),current_q=dict(mean=[0.]*46,std=[1.]*46))
    reports=[]
    for arm in 'ABCD':
        torch.manual_seed(42);m=CampaignPolicy(root/'configs/hd2-reference.yaml',arm,norm,robot).to(device)
        b={f'front_pcd_{i}':torch.rand(bs,1,1,n*6,device=device) for i,n in enumerate([2048,1024,1024],1)}
        b.update({k:torch.rand(bs,1,n,device=device) for k,n in [('state_robot0_joint_pos',22),('state_eef_pose_glass',9),('state_hand_left_qpos',12),('state_hand_right_qpos',12)]})
        b.update(action=torch.rand(bs,64,49,device=device)*.1,action_valid=torch.ones(bs,64,device=device,dtype=torch.bool),dino=torch.rand(bs,16,768,device=device),dino_valid=torch.ones(bs,device=device,dtype=torch.bool),clean_cloud=torch.rand(bs,2048,6,device=device),current_q=torch.rand(bs,46,device=device),current_q_valid=torch.ones(bs,46,device=device,dtype=torch.bool))
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
        start=time.monotonic();before=next(m.parameters()).detach().clone()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):loss,parts=m(b)
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            if arm=='C':assert m.state_readout.head[0].weight.grad.abs().sum()>0
            opt.step()
        assert not torch.equal(before,next(m.parameters()))
        if arm=='C':
            m.eval();x=m.state_readout(m.policy.encoders['front_pcd_1'],b['clean_cloud'])
            for k in b:
                if k.startswith('state_'):b[k]=torch.randn_like(b[k])*100
            y=m.state_readout(m.policy.encoders['front_pcd_1'],b['clean_cloud']);torch.testing.assert_close(x,y)
        reports.append(dict(arm=arm,loss=loss.item(),seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters()),batch=bs,device=device,peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None))
        del m,opt,b
        if device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    print(json.dumps(dict(passed=True,synthetic_only=True,arms=reports),indent=2))
if __name__=='__main__':main()
