"""Independent MuJoCo scenes and verifiers for the eight reasoning families.

Only primitives authored here are used. RGB, instructions, camera calibration
and the answer/view schema are public; instance masks and branched physics are
grader-only. This is a procedural training source, not a benchmark replica.
"""
import hashlib
import json
import math
import random

import numpy as np

try:
    from .efe_pear_curriculum import REASONING_FAMILIES
except ImportError:
    from efe_pear_curriculum import REASONING_FAMILIES

VERSION='rp-jhh-reasoning-mujoco-v1'
WIDTH,HEIGHT=640,512
PALETTE={'red':(.86,.12,.10),'blue':(.10,.30,.88),'green':(.15,.65,.25),
         'yellow':(.93,.78,.10),'purple':(.57,.20,.76),'orange':(.95,.42,.10)}


def camera_xml(name,position,target=(0,0,.04)):
    z=np.array(position,dtype=float)-target;z/=np.linalg.norm(z)
    up=np.array([0,1,0])if abs(z[2])>.99 else np.array([0,0,1])
    x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x)
    vector=lambda a:' '.join(f'{float(v):.12g}'for v in a)
    return f'<camera name="{name}" pos="{vector(position)}" xyaxes="{vector(x)} {vector(y)}" fovy="45"/>'


def checked_point(value):
    if (not isinstance(value,list)or len(value)!=2 or
        any(type(x)not in (int,float)or not math.isfinite(x)or not 0<=x<=1 for x in value)):
        raise ValueError('A point must contain two finite normalized image coordinates')
    return np.array(value,dtype=float)


class SceneEpisode:
    def __init__(self,family,seed,partition='train'):
        import mujoco
        if family not in REASONING_FAMILIES or partition not in {'train','development'}or type(seed)is not int:
            raise ValueError('Unknown reasoning family, split or seed')
        self.family,self.seed,self.partition=family,seed,partition
        salt=f'{VERSION}/{family}/{partition}/{seed}'
        self.rng=random.Random(int(hashlib.sha256(salt.encode()).hexdigest(),16))
        self.view='front';self.view_actions=0;self.max_view_actions=3;self.seen_target=False
        self.done=False;self.renderer=None;self.reference_point=None
        self.objects=[];self.extra=[];self.target=[];self._make_scene()
        xml=('<mujoco model="independent_reasoning"><compiler angle="radian"/>'
            '<option timestep="0.002" gravity="0 0 -9.81"/>'
            '<visual><global offwidth="640" offheight="512"/><quality shadowsize="2048"/></visual>'
            '<worldbody><light pos="0 -1 3" diffuse=".8 .8 .8"/>'
            '<geom name="floor" type="plane" pos="0 0 -.18" size="2 2 .1" rgba=".55 .56 .58 1"/>'
            +(''if family=='physical_prediction'else
               '<geom name="table" type="box" pos="0 0 -.025" size=".65 .45 .025" rgba=".72 .69 .62 1"/>')
            +camera_xml('front',(0,-1.35,1.45))+camera_xml('side',(1.35,-.25,1.35))
            +camera_xml('overhead',(0,0,1.85))
            +''.join(self.extra)+''.join(self._object_xml(i,o)for i,o in enumerate(self.objects))
            +'</worldbody></mujoco>')
        self.xml=xml;self.model=mujoco.MjModel.from_xml_string(xml);self.data=mujoco.MjData(self.model)
        if family=='physical_prediction':
            for _ in range(300):mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        self.ids=[self.model.geom(f'object_{i}').id for i in range(len(self.objects))]
        self.initial_state=np.concatenate([self.data.qpos,self.data.qvel]).copy()
        # Nanometre-scale canonicalization avoids CPU-architecture ULP drift in
        # settled poses changing a reset's identity. XML remains byte-pinned.
        self.scene_sha256=hashlib.sha256(xml.encode()+self.initial_state.round(9).tobytes()).hexdigest()
        if family=='physical_prediction':self._physical_target()
        if family=='cross_view_correspondence':
            i=self.target[0];self.reference_point=self.data.geom_xpos[self.ids[i]].copy()
            self.reference_point[2]+=self.objects[i]['size'][2]
            if not self.visible_point(self.reference_point,'front',self.ids[i])or not self.visible_point(self.reference_point,'side',self.ids[i]):
                raise ValueError('Reference surface is not visible from both cameras')
            self.view='side'
        if family=='affordance_part_selection':self._check_door_mechanism()

    def _make_scene(self):
        rng=self.rng;colors=list(PALETTE);rng.shuffle(colors)
        n=3+self.seed%4
        cells=[(x,y)for y in [-.24,0,.24]for x in [-.42,-.14,.14,.42]];rng.shuffle(cells)
        for i in range(n):
            x,y=cells[i];shape='box'if i%2==0 else'cylinder'
            self.objects.append({'color':colors[i%6],'shape':shape,
                'position':[x+rng.uniform(-.025,.025),y+rng.uniform(-.015,.015),.055],
                'size':[.05,.045,.055]})
        self.target=[rng.randrange(n)]
        target=self.objects[self.target[0]];label=lambda o:o['color']+' '+('box'if o['shape']=='box'else'cylinder')
        self.question='Point to the '+label(target)+'.'
        self.answer_type='point'
        if self.family=='object_pointing'and self.seed%2:
            self.answer_type='bounding_box'
            self.question='Give a tight axis-aligned bounding box around the visible '+label(target)+'. Use normalized [left, top, right, bottom] image coordinates.'
        if self.family=='set_pointing_counting':
            for i,o in enumerate(self.objects):o['color']='red'if i%2==0 else'blue'
            self.target=[i for i,o in enumerate(self.objects)if o['color']=='red']
            self.question='Point once to every red object and give the exact number of red objects. Include all red shapes.'
            self.answer_type='point_set_count'
        elif self.family=='spatial_relations_frames':
            # A geometrically unique referent, independent of the target's color.
            selected=max(range(n),key=lambda i:self.objects[i]['position'][0]);self.target=[selected]
            self.question=('In the workspace frame, +X is right in the front camera, +Y goes away from its front edge, '
                           'and +Z is up. Point to the object with the largest X coordinate. Use the whole object, not a shadow.')
        elif self.family=='valid_placement':
            self.answer_type='placement_point';self.target=[]
            self.footprint=(.045,.035)
            self.extra += ['<geom type="box" pos="0 -.37 .003" size=".57 .006 .003" rgba="0 .8 .3 1"/>',
                '<geom type="box" pos="0 .37 .003" size=".57 .006 .003" rgba="0 .8 .3 1"/>',
                '<geom type="box" pos="-.57 0 .003" size=".006 .37 .003" rgba="0 .8 .3 1"/>',
                '<geom type="box" pos=".57 0 .003" size=".006 .37 .003" rgba="0 .8 .3 1"/>']
            self.question=('Choose a center point on the tabletop for a new axis-aligned box with half-width .045 m '
                'along workspace X and .035 m along Y. Its complete footprint must stay inside the green border '
                '(X in [-.56,.56] m, Y in [-.36,.36] m), with no contact with another object. Workspace X points '
                'right in the front view, Y away, Z up. Answer in normalized coordinates of the current image.')
        elif self.family=='cross_view_correspondence':
            self.objects[self.target[0]]['shape']='box'
            self.question=('The first image has a magenta ring around one exact point on a box top. '
                'Locate that same surface point in the second image, which has no ring. Return a point in the second image.')
        elif self.family=='affordance_part_selection':
            w,h=rng.uniform(.42,.6),rng.uniform(.28,.4)
            ox,oy=rng.uniform(-.36,-.23),rng.uniform(-.02,.13);z=.06
            handle_x=w-.06;handle_z=.55*h
            self.objects=[{'color':'orange','shape':'box','position':[ox+handle_x,oy-.055,z+handle_z],'size':[.065,.018,.018]}]
            self.target=[0]
            self.extra=[f'<body name="door" pos="{ox} {oy} {z}"><joint name="door_hinge" type="hinge" '
                'axis="0 0 1" damping=".5" range="-1.4 1.4" limited="true"/>'
                f'<geom type="box" pos="{w/2} 0 {h/2}" size="{w/2} .025 {h/2}" rgba=".25 .4 .65 1" mass="1"/>'
                f'<geom name="object_0" type="box" pos="{handle_x} -.055 {handle_z}" size=".065 .018 .018" rgba=".95 .42 .10 1" mass=".05"/>'
                f'<geom type="box" pos="{handle_x-.05} -.035 {handle_z}" size=".008 .02 .008" rgba=".4 .4 .4 1"/>'
                f'<geom type="box" pos="{handle_x+.05} -.035 {handle_z}" size=".008 .02 .008" rgba=".4 .4 .4 1"/></body>',
                f'<geom type="box" pos="{ox-.04} {oy} {z+h/2}" size=".025 .04 {h*.55}" rgba=".2 .25 .3 1"/>']
            self.question='Point to the handle that a gripper should grasp to pull this hinged cabinet door open.'
        elif self.family=='physical_prediction':
            for i,o in enumerate(self.objects):
                o['position'][2]=.225;o['dynamic']=True;o['color']='yellow'
                x,y,_=o['position'];color=PALETTE[colors[i]]
                self.extra.append(f'<geom name="support_{i}" type="box" pos="{x} {y} .09" size=".075 .07 .09" rgba="'+
                                  ' '.join(map(str,color))+' 1"/>')
            self.removed_support=rng.randrange(n)
            self.question=(f'All yellow blocks have the same mass and friction. If the {colors[self.removed_support]} '
                'support is removed while everything else is unchanged, which yellow block falls? '
                'Point to that block in the current, pre-removal image.')
        elif self.family=='active_perception_memory':
            tx,ty=rng.uniform(-.13,.13),rng.uniform(.13,.24)
            self.objects=[{'color':'red','shape':'box','position':[tx,ty,.055],'size':[.055,.05,.055]},
                          {'color':'blue','shape':'cylinder','position':[rng.uniform(.30,.42),-.16,.055],'size':[.05,.045,.055]}]
            self.target=[0]
            self.extra=[f'<geom name="occluder" type="box" pos="{tx} {ty-.15} .22" size=".18 .025 .22" rgba=".32 .34 .36 1"/>']
            self.question=('Find the red box behind the opaque divider. You may inspect the side or overhead camera '
                'before answering. Return a point in the latest image. A final answer requires an observation '
                'that actually reveals the target; the allowed view actions do not move objects.')

    def _object_xml(self,index,obj):
        p=' '.join(map(str,obj['position']));color=' '.join(map(str,PALETTE[obj['color']]))
        size=obj['size'];values=size if obj['shape']=='box'else[size[0],size[2]]
        geom=f'<geom name="object_{index}" type="{obj["shape"]}" size="'+ ' '.join(map(str,values))+f'" rgba="{color} 1" mass=".15" friction=".6 .005 .0001"/>'
        if self.family=='affordance_part_selection':
            return ''  # The handle above belongs to the actual hinged door.
        return f'<body name="body_{index}" pos="{p}">'+('<freejoint/>'if obj.get('dynamic')else'')+geom+'</body>'

    def _physical_target(self):
        import mujoco
        branch=mujoco.MjModel.from_xml_string(self.xml);state=mujoco.MjData(branch)
        state.qpos[:]=self.data.qpos;state.qvel[:]=self.data.qvel
        before=self.data.geom_xpos[self.ids,2].copy();gid=branch.geom(f'support_{self.removed_support}').id
        branch.geom_contype[gid]=0;branch.geom_conaffinity[gid]=0
        for _ in range(800):mujoco.mj_step(branch,state)
        mujoco.mj_forward(branch,state)
        fallen=[i for i in range(len(self.ids))if before[i]-state.geom_xpos[self.ids[i],2]>.10]
        if fallen!=[self.removed_support]:raise ValueError('Branched physics lacks the unique queried falling block')
        self.target=fallen

    def _check_door_mechanism(self):
        import mujoco
        branch=mujoco.MjData(self.model);index=self.model.joint('door_hinge').dofadr[0]
        branch.qfrc_applied[index]=.5
        for _ in range(100):mujoco.mj_step(self.model,branch)
        if abs(float(branch.qpos[self.model.joint('door_hinge').qposadr[0]]))<.01:
            raise ValueError('Articulated door failed its mechanical qualification')

    def project(self,point,view=None):
        camera=self.model.camera(view or self.view).id;rotation=self.data.cam_xmat[camera].reshape(3,3)
        local=rotation.T@(np.array(point)-self.data.cam_xpos[camera]);depth=-local[2]
        if depth<=0:raise ValueError('Point is behind the camera')
        f=.5*HEIGHT/math.tan(math.radians(self.model.cam_fovy[camera])*.5)
        return np.array([(f*local[0]/depth+WIDTH/2)/(WIDTH-1),(-f*local[1]/depth+HEIGHT/2)/(HEIGHT-1)])

    def ray(self,point,view=None):
        camera=self.model.camera(view or self.view).id;p=checked_point(point)
        f=.5*HEIGHT/math.tan(math.radians(self.model.cam_fovy[camera])*.5)
        direction=self.data.cam_xmat[camera].reshape(3,3)@np.array([(p[0]*(WIDTH-1)-WIDTH/2)/f,
            -(p[1]*(HEIGHT-1)-HEIGHT/2)/f,-1.]);direction/=np.linalg.norm(direction)
        return self.data.cam_xpos[camera].copy(),direction

    def visible_point(self,point,view,geomid):
        import mujoco
        camera=self.model.camera(view).id;origin=self.data.cam_xpos[camera].copy()
        direction=np.array(point)-origin;distance=np.linalg.norm(direction);direction/=distance
        hit=np.array([-1],dtype=np.int32)
        length=mujoco.mj_ray(self.model,self.data,origin,direction,None,1,-1,hit)
        return hit[0]==geomid and abs(length-distance)<.01

    def render(self,view=None):
        import mujoco
        from PIL import Image
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,HEIGHT,WIDTH)
        self.renderer.disable_segmentation_rendering();self.renderer.update_scene(self.data,camera=view or self.view)
        rgb=Image.fromarray(self.renderer.render().copy())
        self.renderer.enable_segmentation_rendering();self.renderer.update_scene(self.data,camera=view or self.view)
        segmentation=self.renderer.render().copy();self.renderer.disable_segmentation_rendering()
        mask=np.where(segmentation[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM),segmentation[...,0],-1)
        return rgb,mask

    def observation(self):
        from PIL import ImageDraw
        image,self.mask=self.render();images=[image];names=[self.view]
        if self.family=='cross_view_correspondence':
            reference,_=self.render('front');p=self.project(self.reference_point,'front')*[WIDTH-1,HEIGHT-1]
            ImageDraw.Draw(reference).ellipse((p[0]-9,p[1]-9,p[0]+9,p[1]+9),outline=(255,0,255),width=3)
            images=[reference,image];names=['marked_reference_front',self.view]
        if any(np.count_nonzero(self.mask==self.ids[i])>=20 for i in self.target):self.seen_target=True
        if self.family not in {'valid_placement','active_perception_memory'}and any(
            np.count_nonzero(self.mask==self.ids[i])<20 for i in self.target):
            raise ValueError('A required answer object has too few visible pixels')
        cameras=[]
        for view in (['front','side','overhead']if self.family=='active_perception_memory'else list(dict.fromkeys(['front',self.view]))):
            cid=self.model.camera(view).id
            cameras.append({'name':view,'width':WIDTH,'height':HEIGHT,
                'vertical_fov_degrees':float(self.model.cam_fovy[cid]),
                'position_world_m':self.data.cam_xpos[cid].tolist(),
                'camera_to_world_rotation':self.data.cam_xmat[cid].reshape(3,3).tolist()})
        return {'images':images,'instruction':self.question,'public':{
            'instruction':self.question,'sensor_order':names,'sensors':cameras,
            'sensor_template':'RGB image for each name, in sensor_order. Image u increases right and v down; both range 0 to 1.',
            'camera_frame':'Camera +X right, +Y up, optical axis -Z. Extrinsics map camera vectors to the declared workspace frame.',
            'answer_schema':self.schema(),'view_actions_remaining':self.max_view_actions-self.view_actions}}

    def schema(self):
        schema={'answer_type':self.answer_type,'summary':'short answer justification'}
        if self.answer_type=='point_set_count':schema.update(points='list of distinct [u,v] points',count='nonnegative integer')
        elif self.answer_type=='bounding_box':schema['bbox']='[left, top, right, bottom], four normalized image coordinates'
        else:schema['point']='[u,v], two numbers in [0,1] in the latest image'
        if self.family=='active_perception_memory':schema['optional_action']={'action_type':'view','view':['front','side','overhead']}
        return schema

    def _matches(self,point,index):
        p=checked_point(point);x,y=np.rint(p*[WIDTH-1,HEIGHT-1]).astype(int)
        return bool(self.mask[y,x]==self.ids[index])

    def _placement(self,point):
        origin,direction=self.ray(point)
        if direction[2]>=-1e-8:return False
        center=origin+direction*(-origin[2]/direction[2]);x,y=center[:2];sx,sy=self.footprint
        if abs(x)+sx>.56 or abs(y)+sy>.36:return False
        for o in self.objects:
            dx,dy=abs(x-o['position'][0]),abs(y-o['position'][1])
            if o['shape']=='cylinder':
                if max(0,dx-sx)**2+max(0,dy-sy)**2<=(o['size'][0]+.005)**2:return False
            elif dx<=sx+o['size'][0]+.005 and dy<=sy+o['size'][1]+.005:return False
        return True

    def grade(self,action):
        if action.get('answer_type')!=self.answer_type:return 0.
        try:
            if self.answer_type=='bounding_box':
                b=action['bbox']
                if not isinstance(b,list)or len(b)!=4:return 0.
                checked_point(b[:2]);checked_point(b[2:])
                if b[0]>=b[2]or b[1]>=b[3]:return 0.
                ys,xs=np.where(self.mask==self.ids[self.target[0]])
                target=[xs.min()/(WIDTH-1),ys.min()/(HEIGHT-1),xs.max()/(WIDTH-1),ys.max()/(HEIGHT-1)]
                intersection=max(0,min(b[2],target[2])-max(b[0],target[0]))*max(0,min(b[3],target[3])-max(b[1],target[1]))
                union=(b[2]-b[0])*(b[3]-b[1])+(target[2]-target[0])*(target[3]-target[1])-intersection
                return float(union>0 and intersection/union>=.7)
            if self.answer_type=='point_set_count':
                points=action['points']
                if type(action.get('count'))is not int or not isinstance(points,list)or action['count']!=len(self.target)or len(points)!=len(self.target):return 0.
                matches=[next((i for i in self.target if self._matches(p,i)),None)for p in points]
                return float(None not in matches and len(set(matches))==len(self.target))
            point=action['point'];checked_point(point)
            if self.answer_type=='placement_point':return float(self._placement(point))
            if self.family=='cross_view_correspondence':
                return float(self._matches(point,self.target[0])and np.linalg.norm(np.array(point)-self.project(self.reference_point))<=.025)
            if self.family=='active_perception_memory'and not self.seen_target:return 0.
            return float(any(self._matches(point,i)for i in self.target))
        except (ValueError,TypeError,KeyError,IndexError):return 0.

    def step(self,action):
        if self.done:raise ValueError('Episode already ended')
        if action.get('action_type')=='view':
            if (self.family!='active_perception_memory'or self.view_actions>=self.max_view_actions or
                action.get('view')not in {'front','side','overhead'}):raise ValueError('View action is not allowed')
            self.view=action['view'];self.view_actions+=1
            return self.observation(),0.,False
        reward=self.grade(action);self.done=True
        return None,reward,True

    def oracle_answer(self):
        """Qualification-only answer; never passed into policy observations."""
        if self.family=='active_perception_memory'and not self.seen_target:
            return {'action_type':'view','view':'overhead'}
        action={'answer_type':self.answer_type,'summary':'Discarded verifier qualification.'}
        def point(i):
            ys,xs=np.where(self.mask==self.ids[i]);cx,cy=xs.mean(),ys.mean()
            j=np.argmin((xs-cx)**2+(ys-cy)**2)
            return [float(xs[j]/(WIDTH-1)),float(ys[j]/(HEIGHT-1))]
        if self.answer_type=='point_set_count':action.update(points=[point(i)for i in self.target],count=len(self.target))
        elif self.answer_type=='bounding_box':
            ys,xs=np.where(self.mask==self.ids[self.target[0]])
            action['bbox']=[float(xs.min()/(WIDTH-1)),float(ys.min()/(HEIGHT-1)),float(xs.max()/(WIDTH-1)),float(ys.max()/(HEIGHT-1))]
        elif self.family=='valid_placement':
            for x in np.linspace(-.48,.48,15):
                for y in np.linspace(-.28,.28,13):
                    p=self.project([x,y,0]).tolist()
                    if np.all(np.array(p)>=0)and np.all(np.array(p)<=1)and self._placement(p):
                        action['point']=p;return action
            raise ValueError('Placement generator produced no valid complete footprint')
        elif self.family=='cross_view_correspondence':action['point']=self.project(self.reference_point).tolist()
        else:action['point']=point(self.target[0])
        return action

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
