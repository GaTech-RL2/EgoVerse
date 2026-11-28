"""Map original retarget writer order and world EEF poses into hd2 tensor state."""
import json
from pathlib import Path
import mujoco,numpy as np

MODEL=Path('/coc/flash7/czhang883/Documents/SEW-Geometric-Teleop/projects/xhand_teleop/rby1_with_xhand/model_v1.3_xhand_act.xml')
MOUNT=Path('/coc/flash7/czhang883/Documents/aria_gen2_scripts/outputs/rby_mobile_manip/teleop/head_mount.json')

class StateConverter:
    def __init__(self):
        self.m=mujoco.MjModel.from_xml_path(str(MODEL));self.d=mujoco.MjData(self.m)
        self.mount=np.array(json.loads(MOUNT.read_text())['0726']['M']).reshape(4,4)
        names=[*(f'torso_{i}' for i in range(6)),*(f'right_arm_{i}' for i in range(7)),*(f'left_arm_{i}' for i in range(7)),*(f'head_{i}' for i in range(2))]
        self.adr=[self.m.jnt_qposadr[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,k)] for k in names]
    def fk(self,q):
        self.d.qpos[:]=0;self.d.qpos[self.adr]=q[4:26];mujoco.mj_forward(self.m,self.d)
        out=[]
        for name in ('link_head_2','right_eef'):
            b=self.d.body(name);T=np.eye(4);T[:3,:3]=b.xmat.reshape(3,3);T[:3,3]=b.xpos;out.append(T)
        return out
    def convert(self,g,n=None):
        q=g['obs/robot0_joint_pos'][:];n=len(q) if n is None else n;assert 0<n<=len(q)
        base=g['obs/robot_base_pose'][:];eef=g['obs/eef_right_proprio'][:].reshape(-1,4,4)
        def se2(r):
            x,y,a=base[r];c,s=np.cos(a),np.sin(a);T=np.eye(4);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=[x,y];return T
        offsets=[eef[r,:3,3]-(se2(r)@self.fk(q[r])[1])[:3,3] for r in np.linspace(0,len(q)-1,20).astype(int)]
        shift=np.mean(offsets,0);assert np.median(np.linalg.norm(np.array(offsets)-shift,axis=1))<.005
        Tsh=np.eye(4);Tsh[:3,3]=shift;p9=[]
        for r in range(n):
            T=np.linalg.inv(Tsh@se2(r)@self.fk(q[r])[0]@self.mount)@eef[r]
            p9.append(np.r_[T[:3,3],T[:3,0],T[:3,1]])
        raw=g['actions/joint'][:n];action=np.concatenate((raw[:,22:25],raw[:,14:22],raw[:,:14],raw[:,25:]),axis=-1)
        left=g['obs/hand_left_qpos'][:n];right=g['obs/hand_right_qpos'][:n]
        current=np.concatenate((q[:n,4:10],q[:n,24:26],q[:n,17:24],q[:n,10:17],left,right),axis=-1)
        out=dict(action=action.astype(np.float32),state_robot0_joint_pos=q[:n,4:26].astype(np.float32),
          state_eef_pose_glass=np.asarray(p9,np.float32),state_hand_left_qpos=left.astype(np.float32),state_hand_right_qpos=right.astype(np.float32),
          timestamp=g['obs/aria_ts'][:n]/1e9,current_q=current.astype(np.float32),current_q_valid=np.isfinite(current))
        assert out['action'].shape==(n,49) and current.shape==(n,46)
        assert all(np.isfinite(x).all() for x in out.values())
        assert np.allclose(np.diff(out['timestamp']),1/60,rtol=.05,atol=.001)
        return out
