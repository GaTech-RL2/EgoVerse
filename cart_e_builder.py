# H2-G Cart derivative of the frozen hd2 builder.
# Writes original baseline clouds AND independently sampled E global/clean views.
# Original baseline gaze crop is preserved; it never enters E scene selection.
# Generated initial fork on 2026-11-08; subsequent edits are ordinary source edits.
"""(11-03 15:3x hd2, user: "the highest priority is the dense sample around hands ... slightly more strict cropping for all the parts,
for all global and local point clouds ... your local point cloud does not sample around the hands ... the color neutralization seems to
also center at the wrist, which would make it the tool center point ... 0.9 proprio dropout"; answers: crops 1.0 m everywhere, locals
half at their own hand, grey zone 30 cm.) Copy of real_dp3c_tri_gz_vis_hd.py that writes ALL THREE clouds:
  global g_hd2 (2048): hand-dense as hd30t (512 + 512 within TOY_HR 0.30 m of each TCP, 1024 FPS over the rest of the crop) with the
         tighter crop set by the launcher (TOY_LAT 1.0, TOY_GZR 1.0, TOY_ZMAX 2.5);
  locals l_hd2 / l3_hd2 (1024 each, eef coords): TOY_LR (1.0 m) ball around each eef; hd_sampling.hand_dense_local = 512 within
         TOY_HR of that cloud's own TCP + 512 FPS over the rest of the ball;
  grey: colour within TOY_NEUT (0.30 m) of either TCP -> 0.5 (was 0.15 m of either eef origin).
  hdn_hd2 = pool sizes (global R, global L, local R, local L). The eef pose / gaze arrays stay the base build's (p, gp).

(11-02 23:3x: TOY_HR = hand-ball radius, default 0.15 -> ghd_<ep>; the user asked for 0.20 m ("you only keep the forearm points ...
keep the local geometry around hands ... 20 cm radius") -> TOY_HR=0.20 writes ghd20_<ep> / hdn20_<ep>.)
2026-11-02 23:1x HAND-DENSE variant (user 11-02 ~23:05, second priority after the base run) of real_dp3c_tri_gz_vis.py: identical
lift, robot-camera visibility, greying, gaze centre and crops; ONLY the global 2048 sampling differs (hd_sampling.hand_dense_global:
512 right-hand + 512 left-hand points within 0.15 m of each eef, 1024 FPS over the rest of the gaze crop). Writes ghd_<ep>.npy
(+ hdn_<ep>.npy = hand-pool sizes) into the same clouds dir; the locals / eef pose / gaze / camera arrays are the base build's.
Base builder header:
2026-11-02 cart round-3 copy (user 11-02 ~22:40: round-3 cart policy, "tighter lateral cropping, maybe 1.2 m", "gaze ball only like 1 m
or like 1.2 m", locals "keep it 1.5", "predict a virtual human camera and do an intersection at both training and inference time") of
../fridge_tgz/real_dp3c_tri_gz.py with ONLY these changes:
  (1) TOY_VIS=1 (default): right after the lift to robot glass, keep only the human points the ROBOT camera can see (robot rect camera =
      glass @ [R_DR, 0], real KH, frustum + z-buffer with the dense human cloud as occluders) = the sim 10-26 builder's step
      (../sim_gr1_pour/sim_dp3c_tri_vis.py, sim_policy/virtual_view.visible_from); deploy mirrors it with the predicted human camera.
      The gaze hit still uses the full human cloud. Visible / total point counts per row -> vis_<ep>.npy.
  (2) rows: clouds only for the raise-cut rows (TOY_CUTS final cuts json; = release cart_chuye_1102 r2); the VIO-to-EEF Kabsch fit
      still uses the full take, as in the r1 builds.
  (3) no assemble pass here (assemble.py writes the release).
  Crops set by the launcher: TOY_LAT 1.2, TOY_GZR 1.2, TOY_NG 2048; locals unchanged (1.5 m, 1024).
Earlier header:
2026-10-26 TGZ builder (user 10-26: "one more real-world policy" with the current recipe + gaze crop): REAL Aria data, = the unchanged
basket builder ../chuye_basket/toy_dp3c_body_striped.py with the sim 10-26 recipe (../sim_gr1_pour/sim_dp3c_tri_vis.py, minus its
train-time visibility step, which gained nothing in sim) and the 09-28 gaze crop (tmp/abl/build_gazecrop.py):
  obs.aria_pcdc            (12288,) global: 2048 x xyzrgb, robot-glass coords; A4 crop in the robot rect frame (lateral |(x, y)| < TOY_LAT
                                    1.5 m, depth 0.25..TOY_ZMAX 3.0 m) AND within TOY_GZR (1.5 m, user 10-22) of the gaze point
  obs.aria_pcdc_local      (6144,)  right-eef 1.5 m ball, right-eef coords, 1024 points (unchanged)
  obs.aria_pcdc_local_left (6144,)  left-eef 1.5 m ball, left-eef coords, 1024 points (front_pcd_3, as the sim tri builder)
  obs.eef_pose_glass       (9,)     right-eef pos + rot6d in robot glass (unchanged)
  obs.gaze_point_glass     (3,)     the gaze-crop centre in robot glass (diagnostic / later use)
Colour within TOY_NEUT (0.15 m, sim recipe) of either eef -> 0.5 grey, before every crop. Gaze = Aria cloud MPS eye gaze (user 10-26),
general model, 5 Hz: yaw / pitch read from MPS_ROOT/mps_<vrs>_vrs/eye_gaze/general_eye_gaze.csv (projectaria_tools 2.0.0's read_eyegaze
cannot parse this newer header), direction from projectaria_tools' get_eyegaze_point_at_depth, origin = CPF, both -> device -> rect.
Per row: yaw / pitch interpolated between the two MPS samples around the row (<= 0.3 s apart), else the nearest within 0.15 s, else no
gaze. Centre = robust ray-cloud hit (25th percentile along the ray inside a 0.12 / 0.25 / 0.45 rad cone, as build_gazecrop.py), else the
eef midpoint. Also saved per row: tg = the HUMAN rect camera pose in the robot glass frame (training data for the deploy-time human-camera
predictor, as the sim builder's TG). Env: as the basket builder (TOY_H5, TOY_MAN, TOY_VRSD, TOY_LEROBOT, TOY_DST, TOY_MRG, TOY_SHARDS,
TOY_OUTD, EP_STRIDE, EP_OFF) + MPS_ROOT, TOY_NO_ASSEMBLE=1 (stripes only compute; one later pass assembles)."""
import sys, csv, json, os, glob, h5py, numpy as np, torch, mujoco, pyarrow.parquet as pq, pyarrow as pa, shutil
import pandas as pd
from campaign.gaze_preprocessing import causal_gaze_indices, uniform_clean_cloud, project_measured_gaze, fixed_scene_mask
MAX_GAZE_AGE=.3
sys.path.insert(0,"/coc/flash7/czhang883/Documents/aria_gen2_scripts")
from src.export_frames_as_csv import extract_device_pose, invert_rigid
import src.utils.stereo_utils as a2u
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from egomimic.models.adapt3r_3d_encoder import fps_pytorch
NG=int(os.environ.get("TOY_NG","2048")); NL=1024
ZMAX=float(os.environ.get("TOY_ZMAX","3.0")); LAT=float(os.environ.get("TOY_LAT","1.5"))
GZR=float(os.environ.get("TOY_GZR","1.5")); NEUT=float(os.environ.get("TOY_NEUT","0.15"))
MPS_ROOT=os.environ["MPS_ROOT"]
VIS=os.environ.get("TOY_VIS","1")=="1"; CUTS={int(k):int(v) for k,v in json.load(open(os.environ["TOY_CUTS"]))["cuts"].items()}
sys.path.insert(0,os.path.dirname(__file__))
import virtual_view as VV
sys.path.insert(0,os.path.dirname(__file__))
import hd_sampling as HD
HRV=float(os.environ.get("TOY_HR","0.15")); GHD="ghd" if abs(HRV-0.15)<1e-9 else f"ghd{int(round(HRV*100))}"   # 11-02 23:3x user: 20 cm balls -> TOY_HR=0.20, ghd20_<ep>
# 11-03 00:2x user: "don't do wrist, we should do tcp (our retargeting target)" -> TOY_HC=tcp: ball centres = the XHand TCP (index MCP, the
# SEW retargeting target matched to the human finger-MCP centroid) = eef @ (+-0.0265, 0.0065, -0.1077) m (MuJoCo model_v1.3, zero hand pose);
# the 0.15 m greying stays around the eef origin (base recipe). Output ghd<R>t_<ep> (e.g. ghd30t).
HC=os.environ.get("TOY_HC","tcp"); assert HC=="tcp", HC   # hd2: balls AND grey zone at the TCP
LR=float(os.environ.get("TOY_LR","1.0"))                                   # hd2 local ball radius (was 1.5)
GN,LN,L3N,HNN="g_hd2","l_hd2","l3_hd2","hdn_hd2"
TCP_OFF={"R":np.array([0.0265,0.0065,-0.1077,1.0]),"L":np.array([-0.0265,0.0065,-0.1077,1.0])}
CONES=(0.12,0.25,0.45)
if not os.path.exists(os.environ["TOY_MRG"]):
    with h5py.File(os.environ["TOY_MRG"],"w") as _out:
        _og=_out.create_group("data")
        for _sh in sorted(glob.glob(os.environ["TOY_SHARDS"]+"/shard_*.hdf5")):
            with h5py.File(_sh,"r") as _f:
                for _d in _f["data"]:
                    _f.copy(f"data/{_d}",_og)
    print("merged MRG",flush=True)
pr=lambda *a: print(*a,flush=True)
DEV=os.environ.get('E_BUILD_DEVICE','cuda'); rngg=torch.Generator(device=DEV)
MRG=os.environ["TOY_MRG"]
SRC=os.environ["TOY_LEROBOT"]; DST=os.environ["TOY_DST"]
OUTD=os.environ["TOY_OUTD"]; os.makedirs(OUTD,exist_ok=True)
MAN={r[0]:(r[1],r[2]) for r in csv.reader(open(os.environ["TOY_MAN"]),delimiter="\t") if len(r)>=4}
H5=h5py.File(os.environ["TOY_H5"],"r"); EP2DEMO=sorted(H5["data"].keys()); NEP=len(EP2DEMO)
MH=h5py.File(MRG,"r")
M=np.asarray(json.load(open("/coc/flash7/czhang883/Documents/aria_gen2_scripts/outputs/rby_mobile_manip/teleop/head_mount.json"))["0726"]["M"],float).reshape(4,4)
R_DR=np.array([[0.934902,-0.141367,0.325537],[0.149511,0.988760,0.0],[-0.321878,0.048671,0.945529]])
KH=308.052437*224/512; C0=224/512*256.0
T0V=np.eye(4); T0V[:3,:3]=R_DR                                   # robot rect camera -> glass (visibility)
mj=mujoco.MjModel.from_xml_path("/coc/flash7/czhang883/Documents/SEW-Geometric-Teleop/projects/xhand_teleop/rby1_with_xhand/model_v1.3_xhand_act.xml")
md=mujoco.MjData(mj)
def jadr(n): return mj.jnt_qposadr[mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_JOINT,n)]
TORSO=[jadr(f"torso_{i}") for i in range(6)]; HEAD=[jadr(f"head_{i}") for i in range(2)]
RARM=[jadr(f"right_arm_{i}") for i in range(7)]; LARM=[jadr(f"left_arm_{i}") for i in range(7)]
B_H2=mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_BODY,"link_head_2")
B_RE=mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_BODY,"right_eef")
B_LE=mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_BODY,"left_eef")
def fk(q):
    md.qpos[:]=0
    for a,v in zip(TORSO,q[4:10]): md.qpos[a]=v
    for a,v in zip(HEAD,q[24:26]): md.qpos[a]=v
    for a,v in zip(RARM,q[10:17]): md.qpos[a]=v
    for a,v in zip(LARM,q[17:24]): md.qpos[a]=v
    mujoco.mj_forward(mj,md)
    def T(b):
        X=np.eye(4); X[:3,:3]=md.xmat[b].reshape(3,3); X[:3,3]=md.xpos[b]; return X
    return T(B_H2),T(B_RE),T(B_LE)
def se2(x,y,yaw):
    T=np.eye(4); c,s=np.cos(yaw),np.sin(yaw); T[0,0],T[0,1],T[1,0],T[1,1]=c,-s,s,c; T[0,3],T[1,3]=x,y; return T
def rect_T(dp_):
    dc=dp_.get_device_calibration()
    lc=dc.get_camera_calib("slam-front-left"); rc=dc.get_camera_calib("slam-front-right")
    Tl=lc.get_transform_device_camera().inverse(); Tr=rc.get_transform_device_camera().inverse()
    Rl,_=a2u.create_scanline_rectified_cameras(Tl,Tr)
    Tdl=np.asarray(invert_rigid(np.asarray(Tl.to_matrix(),float)),float)
    R4=np.eye(4); R4[:3,:3]=np.asarray(Rl.to_matrix(),float)
    return Tdl@R4
def kab(X,Y):
    Xc,Yc=X-X.mean(0),Y-Y.mean(0); U,_,Vt=np.linalg.svd(Xc.T@Yc); R=Vt.T@U.T
    if np.linalg.det(R)<0: Vt[-1]*=-1; R=Vt.T@U.T
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=Y.mean(0)-R@X.mean(0); return T
def budg(P_,C_,n):
    if P_.shape[0]>=n: i=fps_pytorch(P_[None],n)[0]; return P_[i],C_[i]
    i=torch.randint(0,max(P_.shape[0],1),(n-P_.shape[0],),generator=rngg,device=DEV)
    if P_.shape[0]: return torch.cat([P_,P_[i]],0),torch.cat([C_,C_[i]],0)
    return torch.zeros(n,3,device=DEV),torch.full((n,3),0.5,device=DEV)
def cap(P_,C_,n=16384):
    if P_.shape[0]>n:
        i=torch.randperm(P_.shape[0],generator=rngg,device=DEV)[:n]; return P_[i],C_[i]
    return P_,C_
def gaze_hit(P_rect,o,d):
    """Robust ray-cloud hit in the RECT frame (= build_gazecrop.py). Returns a 3-D point or None."""
    rel=P_rect-o; along=rel@d; fwd=along>0.15
    if int(fwd.sum())<30: return None
    rel_f,along_f=rel[fwd],along[fwd]
    ang=torch.atan2(torch.linalg.norm(rel_f-along_f[:,None]*d,dim=1),along_f)
    for cone in CONES:
        m=ang<cone
        if int(m.sum())>=30: return o+torch.quantile(along_f[m],0.25)*d
    return None
def mps_gaze(vrsn,T_dev_cpf):
    """MPS general eye gaze -> (ts_ns, yaw, pitch) and the device-frame ray for given yaw / pitch."""
    f=f"{MPS_ROOT}/mps_{vrsn[:-4]}_vrs/eye_gaze/general_eye_gaze.csv"
    df=pd.read_csv(f)
    return (df["tracking_timestamp_us"].to_numpy(np.int64)*1000, df["yaw_rads_cpf"].to_numpy(float), df["pitch_rads_cpf"].to_numpy(float))
def gaze_at(t,gts,gy,gp):
    j=int(np.searchsorted(gts,t))
    if 0<j<len(gts) and gts[j]-gts[j-1]<=300_000_000:
        w=(t-gts[j-1])/max(gts[j]-gts[j-1],1); return (1-w)*gy[j-1]+w*gy[j],(1-w)*gp[j-1]+w*gp[j]
    k=int(np.argmin(np.abs(gts-t))) if len(gts) else -1
    if k>=0 and abs(gts[k]-t)<=150_000_000: return gy[k],gp[k]
    return None
uu,vv=np.meshgrid(np.arange(224),np.arange(224))
RX=torch.tensor((uu-C0)/KH,dtype=torch.float32,device=DEV); RY=torch.tensor((vv-C0)/KH,dtype=torch.float32,device=DEV)
Rdr=torch.tensor(R_DR,dtype=torch.float32,device=DEV)
fk_cert=False; blue_hits=0; blue_checks=0
STRIDE=int(os.environ.get("EP_STRIDE",1)); OFF=int(os.environ.get("EP_OFF",0))
pr(f"recipe hd2: global {NG} hand-dense (A4 lat {LAT} m, depth 0.25-{ZMAX} m, gaze ball {GZR} m), locals {NL} x2 hand-dense in {LR} m balls, hand balls {HRV} m at the TCP, grey {NEUT} m at the TCP; robot-camera visibility {VIS}; MPS {MPS_ROOT}")
for ep in range(NEP):
    if ep % STRIDE != OFF: continue
    if os.path.exists(f"{OUTD}/e_complete_{ep:06d}.json"): continue
    dm=EP2DEMO[ep]; g=H5["data"][dm]; tag=str(g.attrs["source_tag"])
    coll,vrsn=MAN[tag]; vrs=f"{os.environ['TOY_VRSD']}/{vrsn}"
    ts=np.asarray(g["obs/aria_ts"][:],np.int64); TF=len(ts); T=min(TF,CUTS[ep])   # raise-cut rows only (r2)
    q26=np.asarray(g["obs/robot0_joint_pos"][:],np.float32); bp=np.asarray(g["obs/robot_base_pose"][:],np.float32)
    eefR=np.asarray(g["obs/eef_right_proprio"][:],np.float64).reshape(-1,4,4)
    eefL=np.asarray(g["obs/eef_left_proprio"][:],np.float64).reshape(-1,4,4)
    mg=MH["data"][dm]
    IMG=mg["obs/aria_image"]; DEP=mg["obs/aria_depth"]
    ds=[]; dsl=[]
    for r in np.linspace(0,TF-1,20).astype(int):
        _,Te,Tl_=fk(q26[r]); W=se2(*bp[r])
        ds.append(eefR[r][:3,3]-(W@Te)[:3,3]); dsl.append(eefL[r][:3,3]-(W@Tl_)[:3,3])
    shift=np.mean(ds,0)
    assert np.median([np.linalg.norm(d-shift) for d in ds])<0.005, f"{dm} R-FK gate"
    assert np.median([np.linalg.norm(d-np.mean(dsl,0)) for d in dsl])<0.02, f"{dm} L-FK gate"
    if not fk_cert: pr(f"FK CERT ok (R<5mm, L<20mm) first shift {shift.round(3)}"); fk_cert=True
    Tsh=np.eye(4); Tsh[:3,3]=shift
    dp_=data_provider.create_vrs_data_provider(vrs); Tdr_h=rect_T(dp_); Trd=np.linalg.inv(Tdr_h)
    T_dev_cpf=np.asarray(dp_.get_device_calibration().get_transform_device_cpf().to_matrix(),float)
    o_dev=T_dev_cpf[:3,3]
    gts,gyaw,gpit=mps_gaze(vrsn,T_dev_cpf)
    gaze_indices,gaze_time_valid,gaze_age=causal_gaze_indices(ts[:T]/1e9,gts/1e9,MAX_GAZE_AGE)
    clean_view=np.zeros((T,2048,6),np.float32); e_global=np.zeros((T,2048,6),np.float32)
    gaze_uv=np.full((T,2),np.nan,np.float32); gaze_valid=np.zeros(T,bool); gaze_timestamp=np.full(T,np.nan)
    e_rng=torch.Generator(device=DEV)
    calibration=dict(camera_frame='human_rect_224',intrinsics=[KH,KH,C0,C0],image_shape=[224,224],
      T_device_rect=Tdr_h.tolist(),T_device_cpf=T_dev_cpf.tolist(),vrs=vrs,
      projection='current_depth_ray_intersection_25th_percentile_cones_0.12_0.25_0.45',
      max_age_seconds=MAX_GAZE_AGE,alignment='latest_observed_mps_general_no_interpolation')
    psid=dp_.get_stream_id_from_label("vio_high_frequency"); hsid=dp_.get_stream_id_from_label("handtracking")
    Xod,Yw=[],[]
    for r in range(0,TF,4):
        sd=dp_.get_sensor_data_by_time_ns(psid,int(ts[r]),TimeDomain.DEVICE_TIME,TimeQueryOptions.CLOSEST)
        Tp,_g,_q=extract_device_pose(sd,"vio_high_frequency")
        if Tp is None: continue
        sdh=dp_.get_sensor_data_by_time_ns(hsid,int(ts[r]),TimeDomain.DEVICE_TIME,TimeQueryOptions.CLOSEST)
        if abs(sdh.get_time_ns(TimeDomain.DEVICE_TIME)-int(ts[r]))/1e6>34: continue
        hp=sdh.hand_pose_data()
        if hp is None: continue
        for one,ee in ((hp.right_hand,eefR[r]),(hp.left_hand,eefL[r])):
            if one is None: continue
            try: w=np.asarray(one.get_wrist_position_device(),float)
            except Exception: continue
            Xod.append((np.asarray(Tp,float)@np.r_[w,1.0])[:3]); Yw.append(ee[:3,3])
    assert len(Xod)>=12, f"{dm} few corr"
    Tfit=kab(np.array(Xod),np.array(Yw))
    G=np.zeros((T,6*NG),np.float32); L=np.zeros((T,6*NL),np.float32); L3=np.zeros((T,6*NL),np.float32)
    P9=np.zeros((T,9),np.float32); GP=np.zeros((T,3),np.float32); TG=np.zeros((T,4,4)); src=np.zeros(T,np.int8); VS=np.zeros((T,2),np.int32); GH=np.zeros((T,6*NG),np.float32); HN=np.zeros((T,4),np.int32); LH=np.zeros((T,6*NL),np.float32); L3H=np.zeros((T,6*NL),np.float32)   # src: 1 hit, 0 fallback
    rngg.manual_seed(ep)
    for r in range(T):
        sd=dp_.get_sensor_data_by_time_ns(psid,int(ts[r]),TimeDomain.DEVICE_TIME,TimeQueryOptions.CLOSEST)
        Tp,_g,_q=extract_device_pose(sd,"vio_high_frequency")
        T_w_rect=Tfit@np.asarray(Tp,float)@Tdr_h
        Th,Te,Tl_=fk(q26[r]); T_w_glass=Tsh@se2(*bp[r])@Th@M
        TG[r]=np.linalg.inv(T_w_glass)@T_w_rect                     # human rect camera in robot glass
        Tg=torch.tensor(TG[r],dtype=torch.float32,device=DEV)        # rect->glass
        dep=torch.tensor(np.asarray(DEP[r],np.float32).reshape(224,224),device=DEV)
        img=torch.tensor(np.asarray(IMG[r],np.float32)/255.0,device=DEV)
        m=dep>0.05
        P=torch.stack([RX[m]*dep[m],RY[m]*dep[m],dep[m]],1)          # rect frame
        C=img[m]
        C[C.sum(1)<0.02]=0.5                                        # colour-invalid -> grey
        # Sensor prefix precedes ALL robot transforms, visibility and TCP colour changes.
        sensor=torch.cat((P,C),dim=1).cpu().numpy()
        clean_view[r]=uniform_clean_cloud(sensor,np.random.default_rng(ep*1000000+r))
        if gaze_time_valid[r]:
            gi=gaze_indices[r]
            direction_cpf=np.asarray(mps.get_eyegaze_point_at_depth(float(gyaw[gi]),float(gpit[gi]),1.0),float)
            direction_rect=Trd[:3,:3]@T_dev_cpf[:3,:3]@direction_cpf
            origin_rect=(Trd@np.append(o_dev,1.))[:3]
            uv,ok=project_measured_gaze(sensor[:,:3],origin_rect,direction_rect,(KH,KH,C0,C0),(224,224))
            if ok:gaze_uv[r]=uv;gaze_valid[r]=True;gaze_timestamp[r]=gts[gi]/1e9
        Pg=P@Tg[:3,:3].T+Tg[:3,3]
        if VIS:                                                     # keep the human points the ROBOT camera can see (before every crop)
            kv=torch.tensor(VV.visible_from(Pg.cpu().numpy(),T0V,KH=KH,C0=C0),device=DEV); VS[r]=(int(kv.sum()),len(kv))
            if int(kv.sum())>=64: Pg,C=Pg[kv],C[kv]
        peR=torch.tensor((np.linalg.inv(T_w_glass)@eefR[r])[:3,3],dtype=torch.float32,device=DEV)
        peL=torch.tensor((np.linalg.inv(T_w_glass)@eefL[r])[:3,3],dtype=torch.float32,device=DEV)
        cR=torch.tensor((np.linalg.inv(T_w_glass)@eefR[r]@TCP_OFF["R"])[:3],dtype=torch.float32,device=DEV)
        cL=torch.tensor((np.linalg.inv(T_w_glass)@eefL[r]@TCP_OFF["L"])[:3],dtype=torch.float32,device=DEV)
        near=(torch.linalg.norm(Pg-cR,dim=1)<NEUT)|(torch.linalg.norm(Pg-cL,dim=1)<NEUT)   # hd2: grey zone at the TCPs
        C=C.clone(); C[near]=0.5                                    # EMBODIMENT COLOUR-EXCISION
        # gaze centre: MPS gaze ray (CPF -> device -> rect) hit on this row's cloud, else the eef midpoint
        hit=None; yp=gaze_at(int(ts[r]),gts,gyaw,gpit)
        if yp is not None:
            d_dev=T_dev_cpf[:3,:3]@np.asarray(mps.get_eyegaze_point_at_depth(float(yp[0]),float(yp[1]),1.0),float)
            o_r=(Trd@np.append(o_dev,1.0))[:3]; d_r=Trd[:3,:3]@d_dev; d_r=d_r/(np.linalg.norm(d_r)+1e-9)
            hit=gaze_hit(P,torch.tensor(o_r,dtype=torch.float32,device=DEV),torch.tensor(d_r,dtype=torch.float32,device=DEV))
        if hit is None: ctr=(peR+peL)/2
        else: ctr=hit@Tg[:3,:3].T+Tg[:3,3]; src[r]=1
        GP[r]=ctr.cpu().numpy()
        # global stream: A4 + depth, AND the gaze ball (the base crop), then the hand-dense sampling
        Pr_=Pg@Rdr; Zr=Pr_[:,2]; slab=(Zr>0.25)&(Zr<ZMAX); ma=slab&(torch.linalg.norm(Pr_[:,:2],dim=1)<LAT)
        mg_=ma&(torch.linalg.norm(Pg-ctr,dim=1)<GZR)
        if int(mg_.sum())<64: mg_=ma
        if int(mg_.sum())<64: mg_=(Zr>0.05)&(Zr<3.0)
        out,sizes=HD.hand_dense_global(Pg,C,cR,cL,slab,mg_,rngg,hr=HRV)
        GH[r]=out.reshape(-1).cpu().numpy(); HN[r,:2]=sizes
        # Separate RNG: E sampling cannot change the paired baseline/local arrays.
        e_slab,e_scene=fixed_scene_mask(Pg.cpu().numpy(),R_DR,LAT,ZMAX)
        e_rng.manual_seed(ep*1000000+r)
        e_out,_=HD.hand_dense_global(Pg,C,cR,cL,torch.as_tensor(e_slab,device=DEV),
          torch.as_tensor(e_scene,device=DEV),e_rng,hr=HRV)
        e_global[r]=e_out.cpu().numpy()
        # hd2 local streams: LR ball around each eef, half of the points within HR of that cloud's own TCP, eef coords
        for pe,ctc,eef,arr,k in ((peR,cR,eefR[r],LH,2),(peL,cL,eefL[r],L3H,3)):
            d=torch.linalg.norm(Pg-pe,dim=1); mb=d<LR
            if int(mb.sum())<64: mb=d<2.5
            outl,szl=HD.hand_dense_local(Pg,C,ctc,mb,rngg,nl=NL,nh=NL//2,hr=HRV); HN[r,k]=szl
            Tge=np.linalg.inv(T_w_glass)@eef; Ti=torch.tensor(np.linalg.inv(Tge),dtype=torch.float32,device=DEV)
            Pe=outl[:,:3]@Ti[:3,:3].T+Ti[:3,3]; arr[r]=torch.cat([Pe,outl[:,3:]],1).reshape(-1).cpu().numpy()
            if r==T//2 and k==2:                                    # gate: grey within NEUT of the right TCP (eef coords)
                nt=torch.linalg.norm(Pe-torch.tensor(TCP_OFF["R"][:3],dtype=torch.float32,device=DEV),dim=1)<NEUT-1e-3
                if int(nt.sum())>0: assert float((outl[nt,3:]-0.5).abs().max())<1e-4, "neutralization failed"
        if r==T//2:                                                 # colour presence in the scene segment
            colr=((out[1024:,3:]-0.5).abs().max(1).values>0.15)
            blue_checks+=1; blue_hits+=int(colr.float().mean()>0.10)
    for nm,arr in (("gaze_hit",src),("gaze_center",GP),("camera_transform",TG),("visibility",VS),(HNN,HN),(LN,LH),(L3N,L3H),(GN,GH)):   # GN last: its presence marks the episode done
        tmp=f"{OUTD}/.{nm}_{ep:06d}.tmp.npy"; np.save(tmp,arr); os.replace(tmp,f"{OUTD}/{nm}_{ep:06d}.npy")
    for nm,arr in [('clean_cloud',clean_view),('e_front_pcd_1',e_global),('gaze_uv',gaze_uv),('gaze_valid',gaze_valid),('gaze_timestamp',gaze_timestamp)]:
        tmp=f"{OUTD}/.{nm}_{ep:06d}.tmp.npy";np.save(tmp,arr);os.replace(tmp,f"{OUTD}/{nm}_{ep:06d}.npy")
    calibration.update(rows=T,gaze_valid_rows=int(gaze_valid.sum()),unique_gaze_observations=int(len(np.unique(gaze_timestamp[gaze_valid]))),
      max_valid_age_seconds=float(np.max(ts[:T][gaze_valid]/1e9-gaze_timestamp[gaze_valid])) if gaze_valid.any() else None,
      clean_cloud_provenance='sensor_uniform_no_pose_no_gaze',e_global_provenance='hd2_hand_dense_fixed_scene_no_gaze',
      training_release_ready=False,projection_verified=False)
    with open(f"{OUTD}/e_complete_{ep:06d}.json","x") as proof:json.dump(calibration,proof,indent=2)
    pr(f"ep{ep} local hand pools median R {int(np.median(HN[:,2]))} L {int(np.median(HN[:,3]))}; >=512: R {float((HN[:,2]>=512).mean()):.2f} L {float((HN[:,3]>=512).mean()):.2f}")
    pr(f"ep{ep} global hand pools median R {int(np.median(HN[:,0]))} L {int(np.median(HN[:,1]))}; >=512: R {float((HN[:,0]>=512).mean()):.2f} L {float((HN[:,1]>=512).mean()):.2f}; <64: R {float((HN[:,0]<64).mean()):.2f} L {float((HN[:,1]<64).mean()):.2f}")
    pr(f"ep{ep} [{tag}] rows {T}, gaze hit {int(src.sum())}/{T}; colourful mid-frames so far {blue_hits}/{blue_checks}")
pr(f"clouds done; COLOURFUL {blue_hits}/{blue_checks} mid-frames")
assert blue_hits>=int(0.8*blue_checks), "colour attach looks broken (clouds are grey)"
pr('stripe done; assemble.py builds the release'); pr('TGZVIS_HD2_STRIPE_DONE')
