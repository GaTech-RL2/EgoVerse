"""Fixed URDF torch FK, named action joints; base deltas deliberately excluded."""
import math
import xml.etree.ElementTree as ET
import torch
from torch import nn

def rpy(v):
    r,p,y=v
    cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
            [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]

class RobotFK(nn.Module):
    def __init__(self,urdf,joint_names,links):
        super().__init__()
        assert len(joint_names)==46 and len(set(joint_names))==46
        assert len(links)==4
        self.joint_names=joint_names;self.links=links
        root=ET.parse(urdf).getroot();self.joints=[];limits={}
        for j in root.findall('joint'):
            origin=torch.eye(4);o=j.find('origin')
            if o is not None:
                origin[:3,:3]=torch.tensor(rpy(list(map(float,o.get('rpy','0 0 0').split()))))
                origin[:3,3]=torch.tensor(list(map(float,o.get('xyz','0 0 0').split())))
            a=j.find('axis');axis=torch.tensor(list(map(float,a.get('xyz','1 0 0').split()))) if a is not None else torch.tensor([1.,0,0])
            axis=axis/axis.norm();idx=len(self.joints)
            self.register_buffer(f'origin_{idx}',origin);self.register_buffer(f'axis_{idx}',axis)
            self.joints.append((j.get('name'),j.get('type'),j.find('parent').get('link'),j.find('child').get('link')))
            l=j.find('limit')
            if l is not None and j.get('type')!='continuous':limits[j.get('name')]=(float(l.get('lower')),float(l.get('upper')))
        allnames={x[0] for x in self.joints};assert set(joint_names)<=allnames
        lo=[];hi=[];limited=[]
        for n in joint_names:
            l,h=limits.get(n,(-1.,1.));assert h>l
            lo.append(l);hi.append(h);limited.append(n in limits)
        for name,value in [('lower',lo),('upper',hi),('width',[h-l for l,h in zip(lo,hi)]),('limited',limited)]:
            self.register_buffer(name,torch.tensor(value))
        self.indices={n:i+3 for i,n in enumerate(joint_names)}
        self.root=next(iter({j[2] for j in self.joints}-{j[3] for j in self.joints}))

    def forward(self,actions):
        shape=actions.shape[:-1];eye=torch.eye(4,device=actions.device,dtype=actions.dtype)
        poses={self.root:eye.expand(*shape,4,4)};pending=list(enumerate(self.joints))
        while pending:
            left=[]
            for i,(name,kind,parent,child) in pending:
                if parent not in poses:left.append((i,(name,kind,parent,child)));continue
                q=actions[...,self.indices[name]] if name in self.indices else actions[...,0]*0
                axis=getattr(self,f'axis_{i}').to(actions);motion=eye.expand(*shape,4,4).clone()
                if kind in ('revolute','continuous'):
                    x,y,z=axis;zero=x*0
                    K=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
                    motion[...,:3,:3]=eye[:3,:3]+q.sin()[...,None,None]*K+(1-q.cos())[...,None,None]*(K@K)
                elif kind=='prismatic':motion[...,:3,3]=q[...,None]*axis
                elif kind!='fixed':raise ValueError(kind)
                poses[child]=poses[parent]@getattr(self,f'origin_{i}').to(actions)@motion
            if len(left)==len(pending):raise ValueError('Disconnected URDF')
            pending=left
        return torch.stack([poses[n] for n in self.links],dim=-3)
