"""Current-sensor-only E views and causal measured-gaze alignment.

All camera transforms/intrinsics are supplied by the recording adapter. Nothing
here estimates eye depth from a fixed constant or uses robot state for gaze.
"""
import numpy as np

def causal_gaze_indices(observation_seconds, gaze_seconds, max_age_seconds=.3):
    obs=np.asarray(observation_seconds, dtype=np.float64)
    gaze=np.asarray(gaze_seconds, dtype=np.float64)
    if not np.isfinite(obs).all() or not np.isfinite(gaze).all() or (np.diff(gaze)<=0).any():
        raise ValueError('Gaze timestamps must be finite, unique and increasing')
    if max_age_seconds<=0: raise ValueError('Positive maximum age required')
    idx=np.searchsorted(gaze,obs,side='right')-1
    if not len(gaze):return np.full(len(obs),-1),np.zeros(len(obs),dtype=bool),np.full(len(obs),np.nan)
    age=obs-gaze[idx.clip(0)]
    valid=(idx>=0)&(age>=0)&(age<=max_age_seconds)
    return np.where(valid,idx,-1),valid,np.where(valid,age,np.nan)

def lift_sensor(depth, rgb, intrinsics):
    """Rectified XYZRGB; no retarget, gaze, TCP, visibility, or hand neutralization."""
    depth=np.asarray(depth);rgb=np.asarray(rgb)
    if rgb.shape!=(*depth.shape,3):raise ValueError('Registered RGB/depth shapes differ')
    fx,fy,cx,cy=intrinsics
    v,u=np.indices(depth.shape)
    keep=np.isfinite(depth)&(depth>.05)
    z=depth[keep]
    xyz=np.stack(((u[keep]-cx)*z/fx,(v[keep]-cy)*z/fy,z),axis=-1)
    colour=rgb[keep].astype(np.float32)/255.
    colour[colour.sum(-1)<.02]=.5
    return np.concatenate((xyz,colour),axis=-1).astype(np.float32)

def uniform_clean_cloud(sensor_points, rng, count=2048):
    if not len(sensor_points):raise ValueError('Empty clean sensor cloud; quarantine the row')
    indices=rng.choice(len(sensor_points),count,replace=len(sensor_points)<count)
    return np.asarray(sensor_points[indices],dtype=np.float32)

def project_measured_gaze(sensor_xyz, origin, direction, intrinsics, image_shape):
    """Current depth supports ray intersection; missing hit => invalid, never EEF fallback."""
    p=np.asarray(sensor_xyz,dtype=np.float64)
    o=np.asarray(origin,dtype=np.float64);d=np.asarray(direction,dtype=np.float64)
    if not np.isfinite(o).all() or not np.isfinite(d).all() or np.linalg.norm(d)<1e-9:
        return np.full(2,np.nan),False
    d=d/np.linalg.norm(d);rel=p-o;along=rel@d;fwd=along>.15
    if fwd.sum()<30:return np.full(2,np.nan),False
    rel,along=rel[fwd],along[fwd]
    angle=np.arctan2(np.linalg.norm(rel-along[:,None]*d,axis=-1),along)
    for cone in (.12,.25,.45):
        hit=angle<cone
        if hit.sum()<30:continue
        point=o+np.quantile(along[hit],.25)*d
        if point[2]<=0:break
        fx,fy,cx,cy=intrinsics;h,w=image_shape
        uv=np.array([(fx*point[0]/point[2]+cx)/w,(fy*point[1]/point[2]+cy)/h])
        if np.isfinite(uv).all() and ((uv>=0)&(uv<=1)).all():return uv.astype(np.float32),True
        break
    return np.full(2,np.nan),False

def fixed_scene_mask(points_glass, rotation_rect_to_glass, lateral=1., zmax=2.5):
    """The E replacement for the global gaze-ball crop; same fixed hd2 slab/lateral rules."""
    rect=np.asarray(points_glass)@np.asarray(rotation_rect_to_glass)
    slab=(rect[:,2]>.25)&(rect[:,2]<zmax)
    mask=slab&(np.linalg.norm(rect[:,:2],axis=-1)<lateral)
    if mask.sum()<64:mask=(rect[:,2]>.05)&(rect[:,2]<3.)
    return slab,mask
