import json
import numpy as np
from campaign.gaze_preprocessing import causal_gaze_indices,lift_sensor,uniform_clean_cloud,project_measured_gaze,fixed_scene_mask

t=np.array([-.01,0.,.1,.199,.2,.51])
idx,valid,age=causal_gaze_indices(t,np.array([0.,.2]),.3)
np.testing.assert_array_equal(idx,[-1,0,0,0,1,-1])
assert (age[valid]>=0).all()
assert not causal_gaze_indices(t,np.array([]))[1].any()
# A future gaze sample never changes earlier alignments.
np.testing.assert_array_equal(causal_gaze_indices(t,np.array([0.,.2,5.]))[0],idx)
depth=np.ones((64,64),dtype=np.float32);rgb=np.full((64,64,3),127,dtype=np.uint8)
intrinsics=(64.,64.,32.,32.)
cloud=lift_sensor(depth,rgb,intrinsics)
clean=uniform_clean_cloud(cloud,np.random.default_rng(42))
assert clean.shape==(2048,6) and np.isfinite(clean).all()
uv,valid=project_measured_gaze(cloud[:,:3],np.zeros(3),np.array([0.,0.,1.]),intrinsics,depth.shape)
assert valid;np.testing.assert_allclose(uv,[.5,.5])
_,valid=project_measured_gaze(cloud[:,:3],np.zeros(3),np.array([0.,0.,-1.]),intrinsics,depth.shape)
assert not valid
_,valid=project_measured_gaze(cloud[:10,:3],np.zeros(3),np.array([0.,0.,1.]),intrinsics,depth.shape)
assert not valid
slab,mask=fixed_scene_mask(cloud[:,:3],np.eye(3));assert slab.all() and mask.all()
print(json.dumps({'passed':True,'causal_no_future_gaze':True,'measured_depth_projection':True,'sensor_only_clean_view':True}))
