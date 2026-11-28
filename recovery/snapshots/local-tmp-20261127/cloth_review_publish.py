import pathlib,json,hashlib,collections,time
R=pathlib.Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
reviewed=[2,4,5,6,7,9,10,12,13,17,18,19,20,21,22,23,24]
plan=json.load(open(R/'plan.json'));accepted=[x for x in plan if (R/'reports/accepted'/(x['uuid']+'.json')).exists()]
assert sorted(x['index']for x in accepted)==reviewed
files={};metrics=[]
for x in accepted:
 for suffix in ['jpg','json']:
  p=R/'reports/visual'/f"{x['index']:02}.{suffix}";files[str(p)]=sha(p)
 metrics.append(json.load(open(R/'reports/visual'/f"{x['index']:02}.json")))
d=dict(passed=True,time=time.time(),reviewed_accepted_uuids=[x['uuid']for x in accepted],files=files,
 scope='Manual visual inspection of first/middle/last RGB, global/right/left clouds and TCP/FK overlays for each of17accepted recovered segments; numeric audit covers all rows separately.',
 findings=['Recovered reaching/grasping, placement and withdrawal actions visible; no gross axis flip or empty cloud in reviewed panels.','Gray robot/hand geometry and sparse local environment points retained as existing hd2 recipe.','All reviewed segments gaze-hit fraction1.0.'],
 limitations=['Three visual samples per segment, not a frame-by-frame visual review.','Existing retarget gate permits brief IK residual outliers: sampled stored-target versus actual FK TCP error reaches88.2mm at take84 recovery start; actual robot-q FK tensors are independently consistent. Original QC thresholds unchanged; first recovered action masked where boundary crosses prior gap.','Local hand neighborhoods may be sparse when hand leaves view; fixed point count is not independent physical coverage.','Take94 additional interval failed original FK gate and has no >1s reliable refinement; its original accepted portion remains.'],
 sampled_metrics=metrics)
p=R/'reports/visual-review.json'
with p.open('x')as f:json.dump(d,f,indent=2)
print('VISUAL_REVIEW_RECORDED',len(accepted))
