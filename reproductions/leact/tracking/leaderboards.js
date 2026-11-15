/* Benchmark-specific result rows. No training loss or cross-benchmark ranks. */
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  if(root)root.LeActLeaderboards=api;
})(typeof window==='undefined'?null:window,function(){
  'use strict';
  const finite=Number.isFinite;
  const fmt=v=>finite(v)?(v!==0&&Math.abs(v)<.0001?v.toExponential(3):v.toLocaleString(undefined,{maximumSignificantDigits:4})):'—';
  function idFor(value){let hash=2166136261;for(const c of value)hash=Math.imul(hash^c.charCodeAt(0),16777619);return 'result-'+(hash>>>0).toString(36);}
  function classify(e){
    const path=e.label||'',checkpoint=e.summary?.checkpoint||'';
    const run=path.split('/')[0],round=path.match(/\/round-(\d+)\/eval\//)?.[1];
    const make=(group,name,kind,stage,legacy=false)=>({group,name,kind,run,stage,legacy});
    if(path.startsWith('frontier-interim-'))return make('frontier-diagnostic','Frontier LeAct · one-seed diagnostic','diagnostic','');
    if(path.startsWith('self-leact-20261024-v1/')&&round)return make('self-leact-'+round,`Self warmstart + LeAct · round ${round}`,'leact',`leact/round-${round}`);
    if(path.startsWith('overnight-20261022-frontier/'))return round
      ?make('frontier-leact-'+round,`Frontier LeAct · round ${round}`,'leact',`leact/round-${round}`)
      :make('frontier-warm','Frontier warmstart','warmstart','frontier_4032_n1');
    if(path.startsWith('projection-sampling-20261023-v3/'))return round
      ?make('projection-leact-'+round,`Projection + LeAct · round ${round}`,'projection-leact',`leact/round-${round}`)
      :make('projection-warm','Projection warmstart','projection','projection_4032_n1');
    const scaling=path.match(/\/(self_(\d+)_n(\d+))\//);
    if(scaling)return make(scaling[1],`Self warmstart · ${Number(scaling[2]).toLocaleString('en-US')} × ${scaling[3]}`,'scaling',scaling[1]);
    if(path==='Original BC · final epoch only')return make('original-bc','Original action-only BC','bc','',true);
    if(checkpoint.includes('/experiment-masked-bc-v1/coldstart/'))return make('corrected-bc','BC · corrected loss mask','bc-corrected','exit/bc',true);
    if(checkpoint.includes('/concise-reproduction-v1/anchor/'))return make('pilot-warm','Small self-generated warmstart','warmstart','anchor',true);
    if(checkpoint.includes('/concise-reproduction-v1/')&&round){
      const exit=checkpoint.includes('/exit/');
      return make(`pilot-${exit?'exit':'leact'}-${round}`,`Earlier ${exit?'ExIt':'LeAct'} · round ${round}`,exit?'exit':'leact',`${exit?'exit/exit':'leact/leact'}/round-${round}`,true);
    }
    return make(path.replace(/\/eval\/.*/,''),path.replace('/collected/latest/',' · '),'other',path.split('/collected/latest/')[1]?.split('/eval/')[0]||'',true);
  }
  function pokerRows(data){
    const byCheckpoint=new Map(),groups=new Map();
    for(const e of data.evaluations||[]){
      if(!finite(e.summary?.single?.exploitability?.mean)||!finite(e.summary?.oracle_bon?.exploitability?.mean))continue;
      const key=e.summary.checkpoint||e.label;
      // A retry copy is not a second trained model. Prefer the fuller seed set.
      if(byCheckpoint.has(key)&&(byCheckpoint.get(key).summary.inference_seeds?.length||0)>=(e.summary.inference_seeds?.length||0))continue;
      byCheckpoint.set(key,e);
    }
    const rows=[...byCheckpoint.values()].map(e=>{
      const c=classify(e),checkpoint=e.summary.checkpoint||e.label;
      const row={...c,id:idFor(checkpoint),checkpoint,source:e.label,
        step:Number(checkpoint.match(/checkpoint-(\d+)$/)?.[1]||0),
        single:e.summary.single.exploitability,bon:e.summary.oracle_bon.exploitability,
        seeds:e.summary.inference_seeds||[],measured:true};
      if(!groups.has(c.group))groups.set(c.group,[]);groups.get(c.group).push(row);return row;
    });
    const lineage=data.status?.training_lineage||[];
    const lineageFor=row=>lineage.find(l=>l.run===row.run&&l.stage===row.stage);
    for(const list of groups.values()){
      const last=list.reduce((a,b)=>a.step>b.step?a:b);
      for(const row of list){
        const training=lineageFor(row);
        row.final=row.kind!=='diagnostic'&&(training?.complete&&(!row.step||row.step===training.total_steps)
          ||!row.step||list.length>=3&&row===last);
        row.defaultVisible=row===last;
        row.rankable=row.final&&row.seeds.length===3&&[0,1,2].every(s=>row.seeds.includes(s));
        row.status=row.kind==='diagnostic'?'One-seed diagnostic':row.final?'Final epoch':row===last?'Evaluation in progress':'Intermediate epoch';
        row.initialCheckpoint=training?.initial_checkpoint;
      }
    }
    const base={id:'result-leduc-base',group:'base',name:'Qwen3-8B · untrained base',kind:'base',
      legacy:false,measured:false,rankable:false,final:false,defaultVisible:true,status:'Baseline score not measured'};
    rows.push(base);
    const primary=group=>groups.get(group)?.find(r=>r.final)||groups.get(group)?.at(-1);
    const checkpointRows=new Map(rows.filter(r=>r.checkpoint).map(r=>[r.checkpoint,r]));
    const descriptions={
      warmstart:'SFT teaches the student to produce explanations and expert actions.',
      leact:'Adds SFT on self-generated reasoning that passed the expert-guided filter.',
      'projection-leact':'Applies LeAct after the rewritten-explanation warmstart.',
      projection:'Rewrites frontier explanations before SFT; solver action targets and warmstart budget stay fixed. Meaning changes were found in the pilot.',
      scaling:'Uses more of the saved self-generated reasoning pool. Three epochs stay fixed, so total optimizer steps increase with data.',
      bc:'Learns expert actions without reasoning; this earlier run used the original loss mask.',
      'bc-corrected':'Corrects the action-only loss mask. This remains an earlier BC recipe, not a matched control for frontier LeAct.',
      exit:'Adds SFT on forward-sampled reasoning and actions from the earlier warmstart.',
      diagnostic:'A one-seed interim check. It is excluded from the three-seed ranking.',
      other:'Recorded checkpoint evaluation; inspect the recipe before making a causal comparison.'};
    for(const row of rows){
      if(row===base){row.explanation='The untouched starting model. A matched online evaluation is still needed before attributing speed or quality gains to training.';continue;}
      let baseline=checkpointRows.get(row.initialCheckpoint),relation='Actual training initialization';
      if(row.group==='projection-warm'){baseline=primary('frontier-warm');relation='Matched original-explanation control';}
      else if(row.group==='self_4032_n1'){baseline=primary('self_128_n1');relation='Smaller state-coverage setting';}
      else if(row.group==='self_4032_n4'){baseline=primary('self_4032_n1');relation='Same states; one retained trace each';}
      else if(row.group==='corrected-bc'){baseline=primary('original-bc');relation='Earlier BC implementation; not fully matched';}
      if(!baseline&&['warmstart','bc','scaling'].includes(row.kind)){baseline=base;relation='Starting model; evaluation missing';}
      row.baselineId=baseline?.id;row.baselineName=baseline?.name;row.baselineRelation=relation;
      row.explanation=descriptions[row.kind]||descriptions.other;
      if(baseline?.single){
        const ratio=row.single.mean/baseline.single.mean;
        row.deltaText=ratio===1?'Same single-policy mean':ratio<1?`${fmt(1/ratio)}× lower single-policy exploitability`:`${fmt(ratio)}× higher single-policy exploitability`;
        row.explanation+=` ${row.deltaText} than the linked baseline.`;
      }
      if(!row.final&&row.kind!=='diagnostic')row.explanation+=' This is an intermediate evaluation, not a final-epoch result.';
      if(row.legacy)row.explanation+=' Earlier pilot protocol.';
    }
    const pending=(group,name,baselineGroup,description)=>{
      if(groups.has(group))return;
      const baseline=primary(baselineGroup)||rows.find(r=>r.group===baselineGroup);
      rows.push({id:'result-'+group,group,name,kind:'pending',legacy:false,measured:false,rankable:false,
        defaultVisible:true,status:'Awaiting evaluation',explanation:description,
        baselineId:baseline?.id,baselineName:baseline?.name,baselineRelation:'Planned comparison'});
    };
    if(data.status?.self_followup||lineage.some(l=>l.stage==='self_4032_n4'))pending('self_4032_n4','Self warmstart · 4,032 × 4','self_4032_n1','Tests whether more retained traces per state improve on the one-trace setting. No completed score has been imported.');
    if(data.status?.projection_sampling)pending('projection-leact-2','Projection + LeAct · round 2','projection-leact-1','The final projection comparison is pending. Its warmstart and first-round measurements remain available above.');
    if(data.status?.self_leact){
      pending('self-leact-1','Self warmstart + LeAct · round 1','self_4032_n4','Applies the matched LeAct settings to the largest self-generated warmstart. No completed round score yet.');
      pending('self-leact-2','Self warmstart + LeAct · round 2','self-leact-1','Second LeAct round after the self-generated warmstart. Warmstart data and compute differ from the frontier arm.');
    }
    return rows;
  }
  function builderRows(data){
    const speed=data.status?.builderbench_speed||{},reports=speed.reports||[];
    const cohort=r=>JSON.stringify([(r.task_metadata||[]).map(t=>[t.id,t.task_file_sha256]).sort(),
      r.planned_rollouts,r.seed_ids||[],r.protocol.max_actions,r.protocol.candidates_per_decision]);
    const mainExpert=reports.find(r=>r.role==='expert'&&!r.source.includes('/expert-probe/'));
    const rows=reports.map(r=>{
      const pilot=r.source.includes('/expert-probe/');
      const scope=pilot?'pilot':!mainExpert||cohort(r)===cohort(mainExpert)?'full':'cohort-'+idFor(cohort(r));
      const name=r.role==='expert'?(pilot?'GPT-6 Astra · capability pilot':'GPT-6 Astra · expert'):r.label;
      const stageId=r.source.split('/').at(-1);
      const checkpoint=(data.status?.builderbench_student?.checkpoints||[]).find(c=>c.id===stageId);
      const row={id:idFor(r.run_sha256),name,role:r.role,scope,runSha:r.run_sha256,stageId,checkpoint:r.checkpoint,
        parentStage:checkpoint?.parent_stage,finalEpoch:checkpoint?.final_epoch,
        source:r.source,measured:true,rankable:r.complete,complete:r.complete,
        progress:r.mean_cube_progress,success:r.success_rate,solved:r.full_successes,
        completed:r.completed_rollouts,planned:r.planned_rollouts,
        taskCount:r.task_metadata.length,scopeLabel:pilot?'Six-task pilot · two rollouts each':`${r.task_metadata.length} tasks · ${r.planned_rollouts} planned rollouts`,decisionSeconds:r.decision_seconds.mean,
        p95:r.decision_seconds.p95,modelSeconds:r.rollout_model_seconds.mean,
        wallSeconds:r.rollout_wall_seconds.mean,status:r.complete?'Complete':'Collection in progress',
        protocol:r.protocol};
      row.explanation=r.role==='expert'
        ?`${r.full_successes}/${r.completed_rollouts} completed rollouts solved the whole task. ${pilot?'A six-task feasibility probe; scores are not comparable with the 51-task collection.':r.complete?'Expert reference for student quality and deployment speed.':'Remaining tasks can change the aggregate; this partial result is not ranked.'}`
        :'Student task success, cube progress and full-response latency are measured together.';
      const live=data.status?.builderbench_expert;
      if(!r.complete&&live?.output===r.source&&live.status==='stopped'){
        row.status='Collection stopped';
        row.explanation=`${r.full_successes}/${r.completed_rollouts} completed rollouts solved the whole task. Collection stopped: ${live.error||'inspect the run record'}. The partial aggregate remains unranked.`;
      }
      row.raw=r;return row;
    });
    const expert=rows.find(r=>r.role==='expert'&&r.scope==='full');
    const roles=[['base_student','Qwen3-8B · untrained base','expert','Starting model; no student evaluation yet.'],
      ['bc_student','Action-only BC','base_student','Imitates the expert actions without reasoning.'],
      ['warmstart_student','Joint warmstart','bc_student','Trains on explanations and actions from the same teacher corpus.'],
      ['leact_student','LeAct · round 1','warmstart_student','Trains on student reasoning admitted by the LeAct filter.'],
      ['exit_student','Matched CoT ExIt','warmstart_student','Compares forward-sampled training with LeAct from the same warmstart.']];
    for(const [role,name,parent,description] of roles){
      if(rows.some(r=>r.role===role))continue;
      rows.push({id:'result-builder-'+role,name,role,parent,scope:'full',measured:false,rankable:false,
        status:data.status?.builderbench_student?'Queued in automatic pipeline':'Awaiting evaluation',explanation:description+' Quality and speedup are not measured yet.'});
    }
    for(const row of rows){
      if(row.role==='expert')continue;
      const parentRole=row.parent||roles.find(r=>r[0]===row.role)?.[2];
      const baseline=rows.find(r=>row.raw?.baseline_run_sha256===r.runSha&&r.runSha)
        ||(['leact_student','exit_student'].includes(row.role)?rows.find(r=>r.stageId===row.parentStage):null)
        ||rows.find(r=>r.scope===row.scope&&r.role===parentRole&&r.finalEpoch)
        ||rows.find(r=>r.scope===row.scope&&r.role===parentRole)||expert;
      row.baselineId=baseline?.id;row.baselineName=baseline?.name;
      row.baselineRelation=row.role==='base_student'?'Expert quality / deployment reference':'Nearest training control';
      const pair=speed.comparisons?.find(c=>c.reference_run_sha256===expert?.runSha&&c.student_run_sha256===row.runSha&&c.complete);
      row.speedup=pair?.all_matched?.model_rollout_speedup;
      row.speedupScope=pair?`${pair.matched_rollouts} matched task/seed pairs`:'Matched speedup pending';
    }
    return rows;
  }
  function build(data){return {leduc:pokerRows(data),builder:builderRows(data)};}

  let currentData,currentBoards,lastHash=null,fromResult=null;
  const dom=id=>document.getElementById(id);
  function element(tag,text,className){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;}
  function linkTo(row,text){const a=element('a',text||row.name,'result-link');a.href='#'+row.id;a.dataset.resultLink=row.id;return a;}
  function score(value){const n=element('div',fmt(value?.mean),'leaderboard-score');if(finite(value?.mean))n.title='Exact mean: '+value.mean;if(finite(value?.std)){const s=element('small','± '+fmt(value.std));s.title='Exact standard deviation: '+value.std;n.append(s);}return n;}
  function nameCell(row){
    const n=element('div');n.append(linkTo(row));
    n.append(element('span',row.status,'badge'+(row.rankable?'':' issue')));
    if(row.step)n.append(element('small','Optimizer step '+row.step));
    if(row.seeds?.length)n.append(element('small',row.seeds.length+' inference seed(s)'));
    if(row.measured&&row.completed!==undefined)n.append(element('small',`${row.completed}/${row.planned} rollouts`));
    if(row.source){const detail=element('details');detail.append(element('summary','Evidence'),element('code',row.source));if(row.checkpoint)detail.append(element('code',row.checkpoint));n.append(detail);}
    return n;
  }
  function baselineCell(row){
    const n=element('div');
    if(row.baselineId){const a=linkTo({id:row.baselineId,name:row.baselineName});a.dataset.fromResult=row.id;n.append(a,element('small',row.baselineRelation||'Comparison reference'));}
    else n.append(element('span',row.role==='expert'?'Expert reference':row.kind==='base'?'Starting reference':'No comparable baseline recorded','muted'));
    return n;
  }
  function createTable(container,headers,rows,values,sortValue,descending=false){
    const wrap=element('div',undefined,'leaderboard-scroll');wrap.tabIndex=0;wrap.setAttribute('role','region');wrap.setAttribute('aria-label',container==='leduc-leaderboard'?'Leduc results table':'BuilderBench results table');
    const table=element('table',undefined,'leaderboard-table'),thead=element('thead'),tr=element('tr');
    for(const title of headers){const th=element('th',title);th.scope='col';tr.append(th);}thead.append(tr);table.append(thead);
    const body=element('tbody');
    const ordered=[...rows].sort((a,b)=>Number(b.rankable)-Number(a.rankable)||Number(b.measured)-Number(a.measured)
      ||(finite(sortValue(a))&&finite(sortValue(b))?(sortValue(a)-sortValue(b))*(descending?-1:1):0)||a.name.localeCompare(b.name));
    let rank=0;
    for(const row of ordered){
      const tr=element('tr');tr.id=row.id;tr.tabIndex=-1;tr.dataset.group=row.group||row.role;tr.dataset.ranked=String(!!row.rankable);tr.dataset.measured=String(!!row.measured);
      tr.classList.toggle('baseline-target',location.hash==='#'+row.id);
      const cells=[row.rankable?String(++rank):'—',...values(row)];
      cells.forEach((v,index)=>{const td=element('td');if(index===0)td.className='rank-cell';if(index===cells.length-1)td.className='explanation-cell';if(index===cells.length-2)td.className='baseline-cell';td.append(v instanceof Node?v:document.createTextNode(v));tr.append(td);});
      body.append(tr);
    }
    if(!ordered.length){const tr=element('tr'),td=element('td','No completed results for this view yet.');td.colSpan=headers.length;tr.append(td);body.append(tr);}
    table.append(body);wrap.append(table);dom(container).replaceChildren(wrap);
  }
  function render(data){
    currentData=data;currentBoards=build(data);
    const allEpochs=dom('leduc-all-epochs').checked,history=dom('leduc-history').checked;
    const metric=dom('leduc-ranking').value;
    const leduc=currentBoards.leduc.filter(r=>(history||!r.legacy)&&(allEpochs||r.defaultVisible));
    const complete=leduc.filter(r=>r.rankable).length;
    dom('leduc-leaderboard-note').textContent=`${complete} completed final-epoch results shown. Ranked by ${metric==='bon'?'oracle best-of-eight':'single-policy'} exploitability, lower is better. Intermediate checkpoints, missing evaluations and one-seed diagnostics are unranked. ± is variation across inference seeds, not separate training runs.`;
    createTable('leduc-leaderboard',['Rank','Run / checkpoint','Single ↓','Oracle BoN-8 ↓','Nearest baseline','What this result means'],leduc,row=>[
      nameCell(row),score(row.single),score(row.bon),baselineCell(row),element('p',row.explanation)
    ],r=>r[metric]?.mean);
    const gate=data.status?.overnight?.astra_gate;
    dom('leduc-reference-note').textContent='Paper references: LeAct 0.016 ± 0.004; action-only BC 0.003 ± 0.001; warmstart 1.351 ± 0.226 (single-policy exploitability). These reference numbers use the authors’ protocol and are not ranked with this reconstruction.'+(gate?` Astra action expert: ${fmt(gate.exploitability)}; ${gate.passed?'passed':'failed'} the ≤${gate.maximum} gate.`:'');
    const scopeSelect=dom('builder-cohort'),oldScope=scopeSelect.value;
    const scopes=new Map([['full','Full task collection']]);
    for(const row of currentBoards.builder)if(!scopes.has(row.scope))scopes.set(row.scope,row.scopeLabel||row.scope);
    if(JSON.stringify([...scopeSelect.options].map(o=>[o.value,o.textContent]))!==JSON.stringify([...scopes])){
      scopeSelect.replaceChildren(...[...scopes].map(([value,label])=>{const o=element('option',label);o.value=value;return o;}));scopeSelect.value=scopes.has(oldScope)?oldScope:'full';
    }
    const scope=scopeSelect.value||'full',builder=currentBoards.builder.filter(r=>r.scope===scope);
    const builderMetric=dom('builder-ranking').value;
    dom('builder-leaderboard-note').textContent=scope==='pilot'
      ?'Six predeclared tasks × two simulator seeds. This feasibility probe has its own ranking and cannot be compared numerically with the full task collection.'
      :'Ranked only after the planned task/seed set is complete. Cube progress measures placed target cubes; full success also requires the end-effector goal. Speedup appears only for completed matched Astra/student comparisons.';
    createTable('builder-leaderboard',['Rank','Run / coverage','Cube progress ↑','Full success ↑','Decision time','Speed vs Astra','Nearest baseline','What this result means'],builder,row=>{
      const timing=element('div',row.measured?fmt(row.decisionSeconds)+' s':'—','leaderboard-score');
      if(row.measured){timing.append(element('small','p95 '+fmt(row.p95)+' s'),element('small','Model / rollout '+fmt(row.modelSeconds)+' s'));}
      const success=element('div',row.measured?`${row.solved}/${row.completed}`:'—','leaderboard-score');if(row.measured)success.append(element('small',fmt(row.success*100)+'%'));
      return [nameCell(row),element('span',fmt(row.progress),'leaderboard-score'),success,timing,
        element('span',finite(row.speedup)?fmt(row.speedup)+'×':row.role==='expert'?'Reference':'Not measured'),baselineCell(row),element('p',row.explanation)];
    },r=>r[builderMetric],true);
    dom('leaderboard-updated').textContent='Results imported '+new Date(data.storage.imported_utc).toLocaleString()+'. Baselines follow experiment lineage or the closest recorded control, never similarity of scores.';
    const expert=data.status?.builderbench_expert;
    dom('status').textContent=`${currentBoards.leduc.filter(r=>r.rankable&&!r.legacy).length} completed Leduc final-epoch results`+(expert?` · BuilderBench expert collection ${expert.status}: ${expert.completed_rollouts}/${expert.planned_rollouts} rollouts`:' · BuilderBench awaiting evaluation');
    dom('updated').textContent=`Latest imported results: ${new Date(data.storage.imported_utc).toLocaleString()} · ${data.status?.active_workflows?.length||0} GPU workflow(s) recorded as active · Refreshes every 15 seconds`;
    if(location.hash!==lastHash){lastHash=location.hash;if(location.hash.startsWith('#result-'))requestAnimationFrame(()=>navigate(location.hash.slice(1),false));}
  }
  function navigate(id,push=true){
    if(!currentBoards)return;
    const row=[...currentBoards.leduc,...currentBoards.builder].find(r=>r.id===id);
    if(!row)return;
    document.querySelector('[data-tab="benchmarks"]').click();
    if(currentBoards.leduc.includes(row)){
      if(row.legacy)dom('leduc-history').checked=true;
      if(!row.defaultVisible)dom('leduc-all-epochs').checked=true;
    }else dom('builder-cohort').value=row.scope;
    if(push&&location.hash!=='#'+id)window.history.pushState(null,'','#'+id);
    lastHash=location.hash;
    render(currentData);
    const target=dom(id);if(!target)return;
    if(fromResult&&fromResult!==id){
      const source=[...currentBoards.leduc,...currentBoards.builder].find(r=>r.id===fromResult);
      const note=dom('baseline-navigation');note.replaceChildren(element('span','Baseline opened. '));
      if(source)note.append(linkTo(source,'Return to '+source.name));note.hidden=!source;
    }else dom('baseline-navigation').hidden=true;
    target.focus({preventScroll:true});target.scrollIntoView({block:'center',behavior:'auto'});
  }
  function init(){
    for(const id of ['leduc-ranking','leduc-all-epochs','leduc-history','builder-cohort','builder-ranking'])dom(id).addEventListener('change',()=>{if(currentData)render(currentData);});
    document.addEventListener('click',event=>{const a=event.target.closest('a[data-result-link]');if(!a)return;event.preventDefault();fromResult=a.dataset.fromResult||null;navigate(a.dataset.resultLink);});
    window.addEventListener('hashchange',()=>{if(location.hash.startsWith('#result-'))navigate(location.hash.slice(1),false);});
  }
  return {build,idFor,render,init};
});
