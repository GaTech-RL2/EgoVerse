const {test}=require('node:test');
const assert=require('node:assert/strict');
const {build}=require('../tracking/leaderboards.js');

function evaluation(stage,step,single,bon=single/10){
  const run='overnight-20261022-frontier';
  return {label:`${run}/collected/latest/${stage}/eval/checkpoint-${step}`,
    summary:{checkpoint:`/study/${run}/${stage}/sft/checkpoints/checkpoint-${step}`,
      single:{exploitability:{mean:single,std:.001}},oracle_bon:{exploitability:{mean:bon,std:.0001}},inference_seeds:[0,1,2]}};
}
function fixture(){
  const warm=evaluation('frontier_4032_n1',756,.15);
  const r1=evaluation('leact/round-1',378,.017);
  const r2=evaluation('leact/round-2',120,.0078);
  return {evaluations:[warm,r1,r2,evaluation('leact/round-2',40,.0096),evaluation('leact/round-2',80,.0069)],
    status:{training_lineage:[
      {run:'overnight-20261022-frontier',stage:'frontier_4032_n1',total_steps:756,complete:true,initial_checkpoint:'Qwen3-8B'},
      {run:'overnight-20261022-frontier',stage:'leact/round-1',total_steps:378,complete:true,initial_checkpoint:warm.summary.checkpoint},
      {run:'overnight-20261022-frontier',stage:'leact/round-2',total_steps:120,complete:true,initial_checkpoint:r1.summary.checkpoint}]}};
}
function expert({pilot=false,complete=false,tasks=51,role='expert',label='teacher',sha='teacher-sha'}={}){
  return {role,label,run_sha256:sha,source:pilot?'/expert-probe/gpt6-v2':'/teacher-gpt6-v1',
    complete,planned_rollouts:tasks,completed_rollouts:complete?tasks:tasks-2,full_successes:10,
    mean_cube_progress:.6,success_rate:.3,task_metadata:Array.from({length:tasks},(_,i)=>({id:'task-'+i,task_file_sha256:'sha'})),
    seed_ids:[0],protocol:{max_actions:200,candidates_per_decision:1},
    decision_seconds:{mean:6,p95:11},rollout_model_seconds:{mean:98},rollout_wall_seconds:{mean:null}};
}

test('default final epoch does not cherry-pick the best measured checkpoint',()=>{
  const rows=build(fixture()).leduc;
  const round=rows.filter(r=>r.group==='frontier-leact-2');
  assert.equal(round.find(r=>r.defaultVisible).step,120);
  assert.deepEqual(round.filter(r=>r.rankable).map(r=>r.step),[120]);
  assert.ok(round.find(r=>r.step===80).single.mean<round.find(r=>r.step===120).single.mean);
});
test('baseline links follow actual initialization, not whichever checkpoint scored best',()=>{
  const data=fixture(),extra=evaluation('leact/round-1',252,.01,.00001);
  data.evaluations.push(extra);
  const rows=build(data).leduc,r2=rows.find(r=>r.group==='frontier-leact-2'&&r.final);
  assert.equal(rows.find(r=>r.id===r2.baselineId).step,378);
  assert.equal(r2.baselineRelation,'Actual training initialization');
  assert.ok(r2.explanation.includes('lower single-policy exploitability'));
});
test('retry copies collapse and single-seed diagnostics stay unranked',()=>{
  const data=fixture();data.evaluations.push(structuredClone(data.evaluations[0]));
  data.evaluations.push({label:'frontier-interim-eval-20261023/collected/latest/eval/step-126',summary:{...data.evaluations[1].summary,checkpoint:'/interim/checkpoint-126',inference_seeds:[0]}});
  const rows=build(data).leduc;
  assert.equal(rows.filter(r=>r.group==='frontier-warm').length,1);
  assert.equal(rows.find(r=>r.kind==='diagnostic').rankable,false);
});
test('self LeAct is separate from warmstart and links to the actual self checkpoint',()=>{
  const data=fixture();data.status.self_leact={scheduler_status:'PENDING'};
  const warm=evaluation('self_4032_n4',0,.028);
  warm.label='overnight-20261022-self/collected/latest/self_4032_n4/eval/checkpoints';
  warm.summary.checkpoint='/study/self/self_4032_n4/sft/checkpoints';
  data.evaluations.push(warm);
  let rows=build(data).leduc;
  const pending=rows.find(r=>r.group==='self-leact-1');
  assert.equal(pending.measured,false);
  assert.equal(rows.find(r=>r.id===pending.baselineId).group,'self_4032_n4');
  assert.equal(rows.find(r=>r.group==='self-leact-2').baselineId,pending.id);
  const trained=evaluation('leact/round-1',300,.02);
  trained.label='self-leact-20261024-v1/collected/latest/leact/round-1/eval/checkpoint-300';
  trained.summary.checkpoint='/study/self-leact/leact/round-1/sft/checkpoints/checkpoint-300';
  data.evaluations.push(trained);
  data.status.training_lineage.push({run:'self-leact-20261024-v1',stage:'leact/round-1',complete:true,total_steps:300,initial_checkpoint:warm.summary.checkpoint});
  rows=build(data).leduc;
  const result=rows.find(r=>r.group==='self-leact-1');
  assert.equal(result.rankable,true);
  assert.equal(result.legacy,false);
  assert.equal(rows.find(r=>r.id===result.baselineId).checkpoint,warm.summary.checkpoint);
});
test('partial BuilderBench collection and full-benchmark student placeholders are not ranked',()=>{
  const data=fixture();data.status.builderbench_speed={reports:[expert()],comparisons:[]};
  const rows=build(data).builder;
  assert.ok(rows.every(r=>!r.rankable));
  assert.equal(rows.find(r=>r.role==='expert').completed,49);
  assert.equal(rows.find(r=>r.role==='leact_student').speedup,undefined);
  data.status.builderbench_expert={output:'/teacher-gpt6-v1',status:'stopped',error:'Transport failed'};
  const stopped=build(data).builder.find(r=>r.role==='expert');
  assert.equal(stopped.status,'Collection stopped');
  assert.ok(stopped.explanation.includes('Transport failed'));
});
test('pilot and different task sets never share the full-collection ranking',()=>{
  const data=fixture();data.status.builderbench_speed={reports:[expert(),expert({pilot:true,tasks:6,complete:true,sha:'pilot'}),expert({tasks:46,complete:true,role:'leact_student',sha:'student'})]};
  const rows=build(data).builder;
  assert.equal(rows.find(r=>r.runSha==='pilot').scope,'pilot');
  assert.notEqual(rows.find(r=>r.runSha==='student').scope,'full');
});
test('every declared baseline anchor resolves and missing baseline measurements remain explicit',()=>{
  const data=fixture();data.status.builderbench_speed={reports:[expert()],comparisons:[]};
  const boards=build(data),rows=[...boards.leduc,...boards.builder],ids=new Set(rows.map(r=>r.id));
  assert.equal(ids.size,rows.length);
  for(const row of rows)if(row.baselineId)assert.ok(ids.has(row.baselineId),row.name);
  const base=rows.find(r=>r.id==='result-leduc-base');assert.equal(base.measured,false);assert.equal(base.single,undefined);
});
