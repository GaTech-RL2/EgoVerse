import json,math
from pathlib import Path
import torch
from campaign.kinematics import RobotFK
from campaign.objectives import fk_loss,state_loss,flow_loss

def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[2];robot=json.loads((root/'assets/robot.json').read_text())
    fk=RobotFK(root/'assets/robot.urdf',robot['joint_names'],robot['links']).double()
    q=torch.zeros(2,3,49,dtype=torch.float64);q[...,3:]=((fk.lower+fk.upper)/2)
    pose=fk(q);assert torch.isfinite(pose).all()
    import pytorch_kinematics as pk
    chain=pk.build_chain_from_urdf((root/'assets/robot.urdf').read_text()).to(dtype=torch.float64)
    named={name:q[...,i+3].reshape(-1) for i,name in enumerate(robot['joint_names'])}
    for name in chain.get_joint_parameter_names():
        if name not in named:named[name]=torch.zeros(6,dtype=torch.float64)
    reference=chain.forward_kinematics(named)
    for i,name in enumerate(robot['links']):
        torch.testing.assert_close(pose[...,i,:,:].reshape(-1,4,4),reference[name].get_matrix(),atol=1e-6,rtol=1e-6)
    valid=torch.ones(2,3,dtype=torch.bool)
    zero,_=fk_loss(fk,q,q,valid);assert abs(zero.item())<1e-9
    pred=q.clone();pred[...,11]+=.1;pred.requires_grad_()
    loss,_=fk_loss(fk,pred,q,valid);loss.backward()
    assert loss>0 and pred.grad[...,11].abs().sum()>0 and pred.grad[...,:3].abs().sum()==0
    # Central difference checks underlying differentiable FK independently of loss casts.
    def f(x):return fk(x)[...,0,0,3].sum()
    z=q.clone().requires_grad_();g=torch.autograd.grad(f(z),z)[0][0,0,11]
    a=q.clone();b=q.clone();a[0,0,11]+=1e-5;b[0,0,11]-=1e-5
    torch.testing.assert_close(g,(f(a)-f(b))/2e-5,atol=1e-5,rtol=1e-4)
    padded=pred.detach().clone();padded[:,2]=1000;mask=valid.clone();mask[:,2]=False
    l1,_=fk_loss(fk,pred.detach(),q,mask);l2,_=fk_loss(fk,padded,q,mask)
    torch.testing.assert_close(l1,l2)
    readout=torch.randn(3,46,requires_grad=True);target=torch.zeros_like(readout);m=torch.zeros_like(readout,dtype=torch.bool)
    target[:]=float('nan');s=state_loss(readout,target,m,torch.zeros(46),torch.ones(46));s.backward()
    assert s==0 and torch.isfinite(readout.grad).all()
    print(json.dumps(dict(passed=True,fk_reference='pytorch_kinematics',identity=True,finite_difference=True,padding=True,empty_state_mask=True)))
if __name__=='__main__':main()
