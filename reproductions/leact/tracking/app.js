'use strict';
let data;
const runPalette=['#3854c7','#087d70','#d55e00','#9236a4','#b42349','#6b6f00','#007ead','#76523c','#5c34dd','#c2438e','#327d22','#af7200','#536878','#8e2430','#00635c','#bd4d2a','#947133','#735f9e','#26779a','#74514e'];
const runColors=new Map();
const chartState={category:'efe-expanded',hidden:new Set(),only:null,selected:null,pinned:false};
const lossCategories=[
  {id:'efe-expanded',name:'Expanded EFE warmstart',current:true,note:'256 successful Astra episodes, 40 tasks. Qwen3.6 35B, 737 planned updates.'},
  {id:'efe-expanded-leact',name:'Expanded EFE LeAct',note:'Student reasoning selected by expert-action likelihood, with 50% warmstart replay. The checkpoint-496 branch and final-737 branch are reported separately.'},
  {id:'efe-curated',name:'EFE · Curated direct control',note:'New successful-execution-filtered corpus with 64K histories. Memory-test updates are discarded and excluded from this curve; native outcomes are training-task diagnostics.'},
  {id:'efe-warmstart',name:'EFE · Earlier warmstarts',note:'Earlier small-corpus warmstarts, learning-rate comparisons and the 32-to-96-update continuation. These use different data from the expanded run.'},
  {id:'efe-leact',name:'EFE · Earlier LeAct runs',note:'Earlier small-corpus LeAct studies. These are separate from the upcoming expanded-corpus LeAct run.'},
  {id:'efe-recovery',name:'EFE · Recovery archive',note:'First expanded-warmstart generation, held after a numerical failure. Only the valid first 21 updates are shown; rejected updates 22–24 remain excluded.'},
  {id:'builderbench',name:'BuilderBench',note:'Action-only BC, frontier reasoning SFT, LeAct and ExIt. Identical logs copied by the training and evaluation collectors appear once; BuilderBench training remains stopped.'},
  {id:'leduc-frontier',name:'Leduc · Frontier warmstart',note:'Frontier warmstart on 4,032 information states and its subsequent LeAct rounds.'},
  {id:'leduc-self',name:'Leduc · Self-generated reasoning',note:'Self-generated data scaling at 128 / 4,032 states, followed by LeAct from the self-generated warmstart.'},
  {id:'leduc-projection',name:'Leduc · Projection sampling',note:'Projection-sampled warmstart and subsequent LeAct rounds.'},
  {id:'leduc-controls',name:'Leduc · BC baselines',note:'Original action-only BC and the corrected loss-mask baseline. Incomplete historical logs are shown as recorded points only.'},
  {id:'leduc-pilots',name:'Leduc · Earlier pilots',note:'Earlier reasoning pilots, retries and ExIt controls. These are preserved separately from the later frontier and scaling studies.'}
];
function runColor(label){
  if(!runColors.has(label)){
    const index=runColors.size;
    runColors.set(label,runPalette[index]||`hsl(${(index*137.508)%360} 65% 38%)`);
  }
  return runColors.get(label);
}
const el=id=>document.getElementById(id);
const show=v=>v===null||v===undefined?'—':typeof v==='object'?JSON.stringify(v):String(v);
const num=v=>typeof v==='number'?v.toLocaleString(undefined,{maximumSignificantDigits:6}):'—';
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=show(text);if(cls)e.className=cls;return e;}
function table(id,headers,rows){const t=node('table'),head=node('tr');headers.forEach(s=>head.append(node('th',s)));t.append(head);rows.forEach(row=>{const tr=node('tr');row.forEach(v=>{const td=node('td');td.append(v instanceof Node?v:document.createTextNode(show(v)));tr.append(td);});t.append(tr);});el(id).replaceChildren(t);}
function badge(text,issue=false){return node('span',text,'badge'+(issue?' issue':''));}
function list(id,values){el(id).replaceChildren(...values.map(s=>node('li',s)));}
function phaseName(phase){
  if(phase.startsWith('anchor/'))return 'Warmstart';
  if(phase.startsWith('frontier_4032_n1/'))return 'Frontier warmstart · 4,032 states';
  const round=phase.match(/round-(\d+)/);
  if(round)return `${phase.startsWith('exit/')?'ExIt':'LeAct'} · Round ${round[1]}`;
  return phase.replace(/\/?sft\/checkpoints$/, '')||phase;
}
function lossHistories(snapshot){
  const result=(snapshot.training||[]).map(t=>({...t,label:'leduc/'+t.id,name:t.label,category:'leduc-controls',
    phase:'Action-only BC',status:'Training complete',history:t.history||[]}));
  const builderCopies=new Set();
  for(const t of [...(snapshot.live_training||[])].sort((a,b)=>a.run.localeCompare(b.run)||a.phase.localeCompare(b.phase))){
    const builder=t.run.startsWith('builderbench-');
    if(builder){
      const key=JSON.stringify([t.phase,t.dataset?.data_sha256,t.history]);
      if(builderCopies.has(key))continue;
      builderCopies.add(key);
    }
    let category='leduc-pilots',name=t.run,phase=phaseName(t.phase),label=t.run+'/'+t.phase;
    if(builder){
      category='builderbench';const arm=t.phase.split('/')[1];
      name=({bc:'Action-only BC',joint:'Frontier reasoning SFT / warmstart','leact-r1':'LeAct · round 1','exit-r1':'ExIt · round 1'})[arm]||arm;
      phase='BuilderBench · Qwen3-8B';label='builderbench/'+arm;
    }else if(t.run.includes('overnight-20261022-frontier')){category='leduc-frontier';name=phase;phase='Frontier reasoning warmstart lineage';}
    else if(t.run.includes('overnight-20261022-self')||t.run.startsWith('self-leact-')){
      category='leduc-self';const match=t.phase.match(/self_(\d+)_n(\d+)/);
      name=match?`Warmstart · ${Number(match[1]).toLocaleString()} states × ${match[2]} trace${match[2]==='1'?'':'s'}`:phase;
      phase='Self-generated reasoning lineage';
    }else if(t.run.startsWith('projection-sampling-')){
      category='leduc-projection';name=t.phase.startsWith('projection_')?'Projection-sampled warmstart':phase;phase='Projection sampling lineage';
    }
    const history=t.history||[];
    result.push({label,name,phase,category,history,steps:history.at(-1)?.total_steps||history.at(-1)?.step,
      status:t.completed?'Training complete':'Incomplete / in progress',source:t.source,
      history_complete:history.every((r,i)=>r.step===i+1)});
  }
  const study=snapshot.status?.experiment_plan?.embodied_first_exam?.dashboard;
  for(const t of study?.training_runs||[]){
    const expanded=t.id==='efe-qwen36-coverage-warmstart';
    const expandedLeact=t.id?.startsWith('efe-qwen36-coverage-leact-');
    const job=expanded||expandedLeact?study.coverage_expansion?.training?.jobs?.find(j=>j.stage===(expanded?'warmstart':t.id.replace('efe-qwen36-coverage-',''))):null;
    const recovery=expanded?study.coverage_expansion?.training_recovery:null;
    const status=job?.status||recovery?.status;
    result.push({label:t.id||t.name,name:expanded?'Expanded Astra warmstart':t.name,phase:t.phase||'EFE student training',
      category:expanded?'efe-expanded':expandedLeact?'efe-expanded-leact':t.id==='efe-qwen36-coverage-warmstart-generation-1'?'efe-recovery':t.category||(/leact/i.test(t.name)?'efe-leact':'efe-warmstart'),history:t.history||[],
      steps:t.planned_steps||t.history?.at(-1)?.total_steps,status:status?workflowLabel(status):(t.status||(t.complete?'Training complete':'In progress'))+(t.priority?' · '+t.priority+' priority':''),
      detail:t.detail,source:t.source,workflow:job?.workflow||t.workflow,
      history_complete:(t.history||[]).every((r,i)=>r.step===i+1)});
  }
  const expandedLeact=study?.coverage_expansion?.training?.jobs?.find(j=>j.stage==='leact-r1');
  if(expandedLeact&&!result.some(r=>r.category==='efe-expanded-leact'))result.push({
    label:'efe-qwen36-coverage-leact-r1',name:'Expanded warmstart + LeAct',category:'efe-expanded-leact',
    phase:'Starts from the final expanded warmstart',history:[],history_complete:true,
    status:workflowLabel(expandedLeact.status),workflow:expandedLeact.workflow});
  const known=new Set(result.map(t=>t.label));
  for(const arm of study?.schedule_comparison?.arms||[]){
    if(known.has(arm.id)||result.some(t=>t.source?.includes(arm.local_results_directory)))continue;
    result.push({label:arm.id,name:arm.name,phase:'Three-pass schedule comparison',category:'efe-warmstart',
      steps:arm.optimizer_steps,status:'Awaiting first logged optimizer update',history:[],history_complete:true});
  }
  return result;
}
function activateView(id){
  document.body.dataset.view=id;
  document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b.dataset.tab===id);if(b.dataset.tab===id)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
  document.querySelectorAll('.view').forEach(s=>s.classList.toggle('hidden',s.id!==id));
  const titles={benchmarks:'Benchmark leaderboards',losses:'Training progress',overview:'Results & analysis',traces:'Student and teacher traces',audit:'Recipe audit',runs:'Workflow history'};
  document.querySelector('main>h1').textContent=titles[id]||'LeAct experiments';
  document.querySelector('main>.sub').textContent=id==='losses'?'Follow the expanded EFE study, from Astra warmstart to LeAct and PEAR.':'Recorded results, matched baselines and the evidence behind each experiment.';
}
function lossRoute(){
  if(!location.hash.startsWith('#loss-')){
    const target=el(location.hash.slice(1)),view=target?.closest('.view');
    if(view){activateView(view.id);let parent=target;while(parent&&parent!==view){if(parent.tagName==='DETAILS')parent.open=true;parent=parent.parentElement;}requestAnimationFrame(()=>target.scrollIntoView({block:'start'}));}
    return;
  }
  const category=location.hash.slice(6);
  if(!lossCategories.some(c=>c.id===category))return;
  if(chartState.category!==category){
    chartState.category=category;chartState.only=null;chartState.hidden.clear();chartState.selected=null;chartState.pinned=false;
    if(category==='efe-expanded'){const archive=el('loss-categories').querySelector('.study-archive');if(archive)archive.open=false;}
    el('metric').value=category==='efe-expanded'?'both':'loss';
    el('chart-smoothing').value=category==='efe-expanded'?'20':'1';
    el('chart-range').value='all';
  }
  activateView('losses');if(data)chart();requestAnimationFrame(()=>window.scrollTo({top:0,behavior:'instant'}));
}
function workflowLabel(status){
  return ({RUNNING:'Running',PENDING:'Queued',WAITING:'Waiting for allocation',COMPLETED:'Completed',WAITING_FOR_WARMSTART:'Waiting for warmstart',COLLECTED:'Results collected'})[status]||String(status||'Not yet started').replaceAll('_',' ').toLowerCase();
}
function pearCheckLabel(value){
  if(value?.result?.success)return 'Passed';
  if(value?.error||value?.phase?.includes('needs_review'))return 'Needs review';
  if(value?.phase==='fresh_process_restart_verification')return 'Verifying checkpoint';
  return workflowLabel(value?.scheduler_status);
}
function currentStudyStatus(){
  const study=data.status?.experiment_plan?.embodied_first_exam?.dashboard,exp=study?.coverage_expansion;
  if(!exp)return;
  const recovery=exp.training_recovery||{},warm=exp.training?.jobs?.find(j=>j.stage==='warmstart');
  const run=lossHistories(data).find(r=>r.category==='efe-expanded'),last=run?.history.at(-1);
  const state=workflowLabel(warm?.status||recovery.status),stamp=recovery.evidence_captured_utc;
  const link=node('a','Open current loss curves');link.href='#loss-efe-expanded';
  el('status').replaceChildren(node('span',`EFE warmstart: ${state.toLowerCase()}. ${last?.step??0} of ${run?.steps??737} updates recorded.`),link);
  const priorities=exp.execution_priorities;
  if(priorities?.primary_deliverable){
    const r1=exp.training?.jobs?.find(j=>j.stage==='leact-from-step496-r1');
    el('status').replaceChildren(node('span',`Priority: LeAct R1 robot evals. ${workflowLabel(r1?.status)}. ${r1?.progress?.detail||''} Target: ${priorities.target_local}.`),link);
  }
  el('status').dataset.state=(warm?.status||recovery.status||'').toLowerCase();
  const format=value=>value?new Date(value).toLocaleString(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'}):'not available';
  el('updated').textContent=`Training records: ${format(stamp)}. Auto-refresh every 15 seconds.`;
  el('updated').title=`Last dashboard fetch: ${new Date().toISOString()}`;
  el('loss-run-state').textContent=state;el('loss-run-state').dataset.state=(warm?.status||recovery.status||'').toLowerCase();
  const leact=exp.training?.jobs?.find(j=>j.stage==='leact-r1'),pear=study.pear_comparison?.plan?.optimizer_update_qualification;
  const branches=exp.branch_runs?.runs||[];
  const branchText=branches.map(r=>{const job=exp.training?.jobs?.find(j=>j.stage===r.id);return `${r.label}: ${workflowLabel(job?.status)}. ${job?.progress?.detail||''}`;}).join(' ');
  const pending=branchText+' '+(leact?.status==='WAITING_FOR_WARMSTART'?'The final-737 LeAct branch still waits for warmstart completion.':`Final-737 LeAct: ${workflowLabel(leact?.status)}.`);
  const pearText=pear?.result?.success?'PEAR’s optimizer check passed.':pear?.error||pear?.phase?.includes('needs_review')?'PEAR’s optimizer check needs review.':pear?.phase==='fresh_process_restart_verification'?'PEAR is verifying its saved checkpoint.':pear?.accumulation?.microbatches?`PEAR’s update check: ${pear.accumulation.microbatches}/${pear.accumulation.total_microbatches} batches.`:pear?.workflow?`PEAR’s optimizer check is ${workflowLabel(pear.scheduler_status).toLowerCase()}.`:'PEAR’s optimizer check is pending.';
  const selection=exp.selection_workers;
  const workerText=selection?.enabled?`Parallel selection: ${(selection.jobs||[]).filter(j=>j.status==='RUNNING').length}/${(selection.jobs||[]).length} additional H100 nodes running. `:'';
  const rescoreJobs=exp.hf_scoring_recovery?.rescore_jobs||[];
  const rescoreText=rescoreJobs.length?`Training scorer recovery: ${rescoreJobs.filter(j=>j.status==='RUNNING').length}/${rescoreJobs.length} H100 nodes running; production R1 optimization follows verified readmission. `:'';
  const checkpoint=exp.performance?.checkpoint_optimization;
  const checkpointText=checkpoint?` Checkpoint-saving validation: ${workflowLabel(checkpoint.status).toLowerCase()}. ${checkpoint.production_enabled?'Enabled in training.':'The current training save path is unchanged.'}`:'';
  el('loss-next-stage').textContent=pending+' '+workerText+rescoreText+(exp.performance?.summary||'')+checkpointText+' '+
    (priorities?.defer_additional_pear_launches?'Further PEAR++ launches are deferred while LeAct R1 training and its frozen 36-task evaluation panel take priority.':pearText);
}
function chart(){
  const histories=lossHistories(data),category=lossCategories.find(c=>c.id===chartState.category)||lossCategories[0];
  const categoryHistories=histories.filter(t=>t.category===category.id);
  for(const option of el('metric').options)option.disabled=!categoryHistories.some(t=>t.history.some(r=>option.value==='both'?Number.isFinite(r.forward_ce)&&Number.isFinite(r.backward_ce):Number.isFinite(r[option.value])));
  if(el('metric').selectedOptions[0]?.disabled)el('metric').value='loss';
  const selectedMetric=el('metric').value,metric='displayValue',metricLabel=el('metric').selectedOptions[0].textContent;
  const smoothing=Number(el('chart-smoothing').value),range=el('chart-range').value;
  const focus=document.activeElement,focusRun=focus?.dataset.run,focusAction=focus?.dataset.action;
  const focusCategory=focus?.dataset.lossCategory;
  const focusPlot=focus?.id==='training-plot';
  histories.forEach(t=>runColor('training/'+t.label));
  const efeColors={'efe-warmstart-initial':'#3854c7','qwen-warm-bb-1e6-p3':'#087d70','qwen-warm-bb-1e5-p3':'#d55e00','qwen-warm-bb-1e4-p3':'#9236a4','efe-leact-r1':'#b42349'};
  const fieldName={forward_ce:'Forward loss',backward_ce:'Backward loss'};
  const curves=categoryHistories.flatMap(t=>(selectedMetric==='both'?['forward_ce','backward_ce']:[selectedMetric]).map(field=>{
    let window=[],previousStep=null;
    const points=t.history.filter(r=>Number.isFinite(r[field])&&Number.isFinite(r.step)).sort((a,b)=>a.step-b.step).map(r=>{
      if(previousStep!==null&&r.step!==previousStep+1)window=[];
      window.push(r[field]);if(window.length>smoothing)window.shift();previousStep=r.step;
      const mean=window.reduce((sum,value)=>sum+value,0)/window.length;
      return {...r,plotValue:r[field],displayValue:mean,meanWindow:window.length};
    });
    const color=category.id==='efe-expanded'?(field==='backward_ce'?'#b45309':'#275dda')
      :selectedMetric==='both'?runColor('training/'+t.label+'/'+field):efeColors[t.label]||runColor('training/'+t.label);
    return {...t,field,originalLabel:t.label,label:t.label+'/'+field,
      name:selectedMetric==='both'?(categoryHistories.length===1?fieldName[field]:t.name+' / '+fieldName[field]):t.name||t.label,
      phase:category.id==='efe-expanded'?'Expanded Astra warmstart':t.phase,color,allPoints:points,points};
  }));
  const newest=Math.max(1,...curves.flatMap(c=>c.points.map(p=>p.step)));
  if(range!=='all')curves.forEach(c=>{c.points=c.points.filter(p=>p.step>newest-Number(range));});
  const categoryButtons=lossCategories.map(c=>{
    const runs=histories.filter(t=>t.category===c.id),button=node('button',undefined,'loss-category');
    button.type='button';button.dataset.lossCategory=c.id;button.setAttribute('aria-pressed',String(c.id===category.id));
    button.append(node('strong',c.name),node('small',c.current?'Current study':c.upcoming&&!runs.some(t=>t.history.length)?runs[0]?.status||'Awaiting first update':`${runs.filter(t=>t.history.length).length} recorded runs`));
    button.onclick=()=>{location.hash='loss-'+c.id;};return button;
  });
  const studySelect=el('loss-study-select');
  if(!studySelect.options.length)studySelect.replaceChildren(...lossCategories.map(c=>{const option=node('option',c.name);option.value=c.id;return option;}));
  studySelect.value=category.id;
  const archiveOpen=el('loss-categories').querySelector('.study-archive')?.open;
  const historical=node('details',undefined,'study-archive'),heading=node('summary','Earlier experiments');
  historical.open=!category.id.startsWith('efe-expanded')||!!archiveOpen;historical.append(heading,...categoryButtons.slice(2));
  el('loss-categories').replaceChildren(...categoryButtons.slice(0,2),historical);
  el('training-title').textContent=category.name;
  el('loss-category-note').textContent=category.note;
  const current=el('loss-current-run');current.hidden=category.id!=='efe-expanded';
  el('loss-run-state').hidden=category.id!=='efe-expanded';
  if(!current.hidden){
    const run=categoryHistories[0],last=run?.history.at(-1),recent=run?.history.slice(-20)||[];
    const recovery=data.status?.experiment_plan?.embodied_first_exam?.dashboard?.coverage_expansion?.training_recovery||{};
    const values=node('dl',undefined,'current-values');
    for(const [label,value] of [['Recorded updates',`${last?.step??0} / ${run?.steps??737}`],['Forward · last 20',recent.length?(recent.reduce((s,r)=>s+r.forward_ce,0)/recent.length).toFixed(3):'—'],['Backward · last 20',recent.length?(recent.reduce((s,r)=>s+r.backward_ce,0)/recent.length).toFixed(3):'—'],['Saved checkpoint',recovery.committed_restart_step??'—']]){
      const item=node('div');item.append(node('dt',label),node('dd',value));values.append(item);
    }
    const progress=node('progress');progress.max=run?.steps||737;progress.value=last?.step||0;progress.setAttribute('aria-label','Recorded warmstart updates');
    current.replaceChildren(values,progress);
  }
  el('loss-metric-note').textContent=category.id.startsWith('efe-')
    ?'EFE loss averages complete-target cross entropy per example. Forward loss includes reasoning and the action; backward loss predicts the explanation given the expert action. These are training losses, not held-out validation losses.'
    :'These runs log total training loss. Separate forward/backward losses were not recorded. Different targets and data make loss values unsuitable as a benchmark leaderboard.';
  table('loss-latest',['Run / metric','Latest raw value','Step / budget','State','Learning rate'],curves.map(c=>{
    const latest=c.allPoints.at(-1),last=c.history.at(-1),label=node('span',c.name);const dot=node('i',undefined,'dot');dot.style.backgroundColor=c.color;label.prepend(dot,document.createTextNode(' '));
    if(c.source)label.title=c.source;
    return[label,latest?num(latest.plotValue):'No recorded value',last?`${last.step} / ${c.steps??'—'}`:`0 / ${c.steps??'—'}`,c.status,last?num(last.learning_rate??last.lr):'—'];
  }));
  el('chart-download').onclick=()=>{
    const csv=[['run','metric','step','raw_value','displayed_mean','mean_window'],...curves.flatMap(c=>c.allPoints.map(p=>[c.originalLabel,c.field,p.step,p.plotValue,p.displayValue,p.meanWindow]))]
      .map(row=>row.map(value=>'"'+String(value).replaceAll('"','""')+'"').join(',')).join('\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),link=node('a');
    link.href=url;link.download=category.id+'-losses.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  const isVisible=c=>(!chartState.only||chartState.only===c.label)&&!chartState.hidden.has(c.label);
  const visible=curves.filter(c=>isVisible(c)&&c.points.length),all=visible.flatMap(c=>c.points);
  const groups=new Map(),items=new Map();
  let guide,marker;

  function highlight(label){
    groups.forEach((group,key)=>{
      group.setAttribute('opacity',label&&key!==label?'.16':'1');
      group.querySelector('.trend-curve')?.setAttribute('stroke-width',key===label?'3':'2.3');
    });
    items.forEach((item,key)=>item.classList.toggle('is-highlighted',key===label));
  }
  function clearSelection(){
    chartState.selected=null;chartState.pinned=false;
    guide?.setAttribute('visibility','hidden');marker?.setAttribute('visibility','hidden');
    el('chart-readout').replaceChildren(node('p','Point to a curve to see its run, optimizer step and exact value.','muted'));
    delete el('chart-readout').dataset.run;delete el('chart-readout').dataset.step;
    el('chart-clear').disabled=true;el('chart-announcement').textContent='';highlight(null);
  }
  el('chart-clear').onclick=clearSelection;
  el('chart-show-all').onclick=()=>{
    chartState.hidden.clear();chartState.only=null;clearSelection();chart();
  };
  el('chart-run-count').textContent=`${visible.length} of ${curves.length} curves shown`;
  el('chart-show-all').disabled=curves.every(isVisible);

  el('legend').replaceChildren(...curves.map(c=>{
    const item=node('div',undefined,'legend-item'),toggle=node('button',undefined,'run-toggle');
    const dot=node('i',undefined,'dot'),label=node('span',undefined,'legend-label');
    dot.style.backgroundColor=c.color;label.append(node('span',c.name));
    if(c.phase)label.append(node('small',c.phase));
    label.append(node('small',c.status));
    if(!c.points.length)label.append(node('small',c.history.length?'This metric was not logged':'No optimizer updates logged yet'));
    else if(!c.history_complete)label.append(node('small','Partial history · recorded points only'));
    toggle.type='button';toggle.dataset.run=c.label;toggle.dataset.action='toggle';
    toggle.setAttribute('aria-pressed',String(isVisible(c)));
    toggle.setAttribute('aria-label',`${isVisible(c)?'Hide':'Show'} ${c.label}`);
    toggle.title=c.label;toggle.append(dot,label);
    toggle.onclick=()=>{
      const shown=new Set(curves.filter(isVisible).map(t=>t.label));
      if(shown.has(c.label))shown.delete(c.label);else shown.add(c.label);
      chartState.only=null;chartState.hidden=new Set(curves.filter(t=>!shown.has(t.label)).map(t=>t.label));
      chart();
    };
    const only=node('button','Only','run-only');
    only.type='button';only.dataset.run=c.label;only.dataset.action='only';only.disabled=!c.points.length;
    only.setAttribute('aria-label',`Show only ${c.label}`);
    only.onclick=()=>{chartState.only=c.label;chartState.hidden.delete(c.label);chart();};
    item.classList.toggle('is-hidden',!isVisible(c));item.append(toggle,only);items.set(c.label,item);
    const preview=()=>{
      if(isVisible(c)&&c.points.length&&groups.has(c.label)&&!chartState.pinned){
        const step=chartState.selected?.step??c.points.at(-1).step;
        const point=c.points.reduce((best,p)=>Math.abs(p.step-step)<Math.abs(best.step-step)?p:best,c.points[0]);
        inspect(c,point);
      }else highlight(isVisible(c)?c.label:chartState.selected?.label);
    };
    const restore=()=>highlight(chartState.selected?.label);
    item.addEventListener('pointerenter',preview);item.addEventListener('pointerleave',restore);
    item.addEventListener('focusin',preview);
    item.addEventListener('focusout',event=>{if(!item.contains(event.relatedTarget))restore();});
    return item;
  }));
  const restoreFocus=()=>{
    if(focusPlot)el('training-plot')?.focus({preventScroll:true});
    else if(focusCategory)el('loss-categories').querySelector(`[data-loss-category="${focusCategory}"]`)?.focus({preventScroll:true});
    else if(focusRun&&focusAction){
      const item=items.get(focusRun);
      item?.querySelector(focusAction==='only'?'.run-only':'.run-toggle')?.focus({preventScroll:true});
    }
  };
  if(!all.length){
    el('chart').replaceChildren(node('p',curves.some(c=>c.points.length)?'All runs with values are hidden. Choose a run below or Show all runs.':'No optimizer updates have been logged for this category and metric yet. Registered runs and their status remain listed below.','chart-empty'));
    clearSelection();el('curve-note').textContent='';restoreFocus();return;
  }

  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
  const width=Math.max(320,Math.min(1280,el('chart').clientWidth)),right=width-24;
  svg.setAttribute('viewBox',`0 0 ${width} 330`);svg.id='training-plot';svg.setAttribute('tabindex','0');
  svg.setAttribute('role','group');svg.setAttribute('aria-label',`${metricLabel} by optimizer step. Use arrow keys to inspect recorded points.`);
  svg.setAttribute('aria-describedby','chart-help');
  const maxX=Math.max(1,...all.map(r=>r.step)),minX=range==='all'?0:Math.max(0,maxX-Number(range));
  const values=all.flatMap(r=>[r.plotValue,r.displayValue]),low=Math.min(...values),high=Math.max(...values),padding=Math.max((high-low)*.06,Math.abs(high)*.02,1e-12);
  const maxY=high+padding,minY=range==='all'?Math.min(0,low-padding):low-padding;
  const x=v=>65+(v-minX)/Math.max(1,maxX-minX)*(right-65),y=v=>275-(v-minY)/(maxY-minY)*245;
  const axis=value=>value===0?'0':Math.abs(value)<.001?value.toExponential(1):Number(value.toPrecision(3)).toLocaleString();
  function s(tag,attrs,text,parent=svg){
    const n=document.createElementNS(ns,tag);
    Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));
    if(text!==undefined)n.textContent=text;parent.append(n);return n;
  }
  s('rect',{x:65,y:30,width:right-65,height:245,fill:'transparent'});
  for(let i=0;i<=4;i++){
    const val=minY+(maxY-minY)*i/4;
    s('line',{x1:65,y1:y(val),x2:right,y2:y(val),stroke:'#e3ebe8'});
    s('text',{x:55,y:y(val)+4,'text-anchor':'end',fill:'#526580','font-size':12},axis(val));
  }
  for(const val of new Set(Array.from({length:4},(_,i)=>Math.round(minX+(maxX-minX)*i/3))))
    s('text',{x:x(val),y:300,'text-anchor':'middle',fill:'#526580','font-size':12},String(val));
  s('text',{x:width/2,y:324,'text-anchor':'middle',fill:'#526580','font-size':12},'Optimizer update');
  s('text',{x:65,y:16,fill:'#526580','font-size':12},selectedMetric==='both'?'Cross entropy':metricLabel);
  visible.forEach(c=>{
    const group=s('g',{'data-run':c.label,class:'chart-curve'});groups.set(c.label,group);
    if(c.history_complete&&smoothing>1)s('polyline',{points:c.points.map(p=>`${x(p.step)},${y(p.plotValue)}`).join(' '),fill:'none',stroke:c.color,'stroke-width':1,opacity:'.22',class:'raw-curve'},undefined,group);
    if(c.history_complete)s('polyline',{points:c.points.map(p=>`${x(p.step)},${y(p[metric])}`).join(' '),fill:'none',stroke:c.color,'stroke-width':2.3,'stroke-dasharray':c.field==='backward_ce'?'7 4':'none',class:'trend-curve'},undefined,group);
    c.points.forEach(p=>s('circle',{cx:x(p.step),cy:y(p[metric]),r:c.history_complete?1.3:4,fill:c.color,'data-step':p.step},undefined,group));
  });
  guide=s('line',{y1:30,y2:275,stroke:'#607780','stroke-dasharray':'4 4','pointer-events':'none',visibility:'hidden',class:'chart-guide'});
  marker=s('circle',{r:6,fill:'white','stroke-width':3,'pointer-events':'none',visibility:'hidden',class:'chart-marker'});
  el('chart').replaceChildren(svg);

  function inspect(curve,point,announce=false){
    chartState.selected={label:curve.label,step:point.step};
    guide.setAttribute('x1',x(point.step));guide.setAttribute('x2',x(point.step));guide.setAttribute('visibility','visible');
    marker.setAttribute('cx',x(point.step));marker.setAttribute('cy',y(point[metric]));
    marker.setAttribute('stroke',curve.color);marker.setAttribute('visibility','visible');highlight(curve.label);
    const heading=node('div',undefined,'inspection-run'),dot=node('i',undefined,'dot');dot.style.backgroundColor=curve.color;
    const label=node('div');label.append(node('strong',curve.name));if(curve.phase)label.append(node('small',curve.phase));
    heading.append(dot,label,node('span',chartState.pinned?'Pinned':'Inspecting','inspection-state'));
    const values=node('dl',undefined,'inspection-values');
    for(const [name,value] of [['Update',point.step],['Raw '+(fieldName[curve.field]||metricLabel).toLowerCase(),point.plotValue],...(smoothing>1?[[`${point.meanWindow}-update mean`,point.displayValue]]:[]),...(Number.isFinite(point.epoch)?[['Epoch',point.epoch]]:[])]){
      const field=node('div');field.append(node('dt',name),node('dd',String(value)));values.append(field);
    }
    el('chart-readout').replaceChildren(heading,values);el('chart-readout').dataset.run=curve.label;
    el('chart-readout').dataset.step=point.step;el('chart-clear').disabled=false;
    if(announce)el('chart-announcement').textContent=`${chartState.pinned?'Pinned. ':''}${curve.name}. Update ${point.step}. Raw value: ${point.plotValue}.`;
  }
  function nearest(event){
    const transform=svg.getScreenCTM();if(!transform)return null;
    const position=new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse());
    if(position.x<65||position.x>right||position.y<30||position.y>275)return null;
    let best=null,distance=Infinity;
    for(const curve of visible)for(const point of curve.points){
      const d=(x(point.step)-position.x)**2+(y(point[metric])-position.y)**2;
      if(d<distance){distance=d;best={curve,point};}
    }
    return best;
  }
  svg.addEventListener('pointermove',event=>{
    if(chartState.pinned||event.pointerType==='touch')return;
    const found=nearest(event);
    if(found&&(chartState.selected?.label!==found.curve.label||chartState.selected?.step!==found.point.step))inspect(found.curve,found.point);
  });
  // Click coordinates can be rounded to pixels; preserve the precise point
  // selected on pointer-down when several curves are very close together.
  let pressedPoint=null;
  svg.addEventListener('pointerdown',event=>{pressedPoint=nearest(event);});
  svg.addEventListener('pointercancel',()=>{pressedPoint=null;});
  svg.addEventListener('click',event=>{
    const found=pressedPoint||nearest(event);pressedPoint=null;if(!found)return;
    const same=chartState.selected?.label===found.curve.label&&chartState.selected?.step===found.point.step;
    chartState.pinned=!(chartState.pinned&&same);inspect(found.curve,found.point,true);
  });
  svg.addEventListener('keydown',event=>{
    if(event.key==='Escape'){event.preventDefault();clearSelection();return;}
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','Enter',' '].includes(event.key))return;
    event.preventDefault();
    let curveIndex=Math.max(0,visible.findIndex(c=>c.label===chartState.selected?.label));
    if(event.key==='ArrowUp')curveIndex=(curveIndex+visible.length-1)%visible.length;
    if(event.key==='ArrowDown')curveIndex=(curveIndex+1)%visible.length;
    const curve=visible[curveIndex],step=chartState.selected?.step??curve.points[0].step;
    let index=curve.points.reduce((best,p,i)=>Math.abs(p.step-step)<Math.abs(curve.points[best].step-step)?i:best,0);
    if(event.key==='ArrowLeft')index=Math.max(0,index-1);
    if(event.key==='ArrowRight')index=Math.min(curve.points.length-1,index+1);
    if(event.key==='Home')index=0;if(event.key==='End')index=curve.points.length-1;
    if(event.key==='Enter'||event.key===' ')chartState.pinned=!chartState.pinned;
    inspect(curve,curve.points[index],true);
  });
  const selected=visible.find(c=>c.label===chartState.selected?.label);
  const point=selected?.points.find(p=>p.step===chartState.selected.step);
  if(point)inspect(selected,point);else clearSelection();
  el('curve-note').textContent=`${all.length.toLocaleString()} recorded values shown. ${smoothing>1?'Strong lines show the trailing mean of up to 20 consecutive updates; faint lines retain raw values.':'Lines show raw recorded values.'} ${range!=='all'?'Axes fit this update range. ':''}Partial histories remain individual points. Hover values and the table report exact raw measurements.`;
  restoreFocus();
}
function audit(){const issues=new Set(['difference','mismatch','unresolved_assumption','implementation_error','partial']);const rows=(data.audit.rows||[]).filter(r=>el('audit-filter').value==='all'||issues.has(r.status));table('audit-table',['Setting','Paper','Executed / configured','Assessment'],rows.map(r=>{const name=node('div',r.parameter);name.append(node('small',r.source_section));const assessment=node('div');assessment.append(badge(r.status.replaceAll('_',' '),issues.has(r.status)));if(r.note)assessment.append(node('small',r.note));return[name,show(r.paper),show(r.actual),assessment];}));}
function render(){el('status').textContent=data.status.status||'No status recorded';el('updated').textContent=`Experiment status: ${data.status.as_of_utc||'unknown'} · Last changed snapshot: ${data.storage.imported_utc} · ${data.status.active_workflows?.length||0} active reproduction workflows`;const latest=data.training.at(-1),live=data.live_training?.at(-1),step=live?.history.at(-1),phases=Object.values(data.status.reproduction_progress||{}),phase=phases.at(-1)?.phase||'Awaiting allocation';const cards=[['Optimizer steps',step?.step??latest?.steps,live?live.phase:'Corrected BC baseline'],['Latest loss',step?.loss??latest?.history.at(-1)?.loss,'Recorded training loss'],['SFT examples',live?.dataset?.sft_examples??'—',live?.completed?'Training complete':'Current reasoning training stage'],['Reproduction',data.status.numerical_reproduction_complete?'Complete':data.comparisons?.some(c=>c.all_planned_rounds_complete)?'Evaluated':data.status.active_workflows?.length?'In progress':'Blocked',phase.replaceAll('_',' ')]];el('cards').replaceChildren(...cards.map(([label,value,detail])=>{const n=node('div',undefined,'card');n.append(node('div',label,'label'),node('div',typeof value==='number'?num(value):value,'value'),node('div',detail,'detail'));return n;}));chart();table('gates',['Model','Candidates','Usable','Truncated','Positive selected'],data.training.map(t=>[t.label,t.gate?.candidates,t.gate?.usable_candidates,t.gate?.truncated_candidates,t.gate?.positive_selected]));table('evaluations',['Measurement','Single exploitability','Oracle BoN-8','Checkpoint selection'],data.evaluations.filter(e=>e.summary).map(e=>[e.label,`${num(e.summary.single?.exploitability?.mean)} ± ${num(e.summary.single?.exploitability?.std)}`,`${num(e.summary.oracle_bon?.exploitability?.mean)} ± ${num(e.summary.oracle_bon?.exploitability?.std)}`,e.selection]));el('audit-verdict').textContent=data.audit.verdict||'';const st=data.audit.steps;el('step-equation').textContent=st?`${st.examples} examples ÷ (${st.gpus} GPUs × ${st.per_device_batch} per GPU × ${st.gradient_accumulation} accumulation) × ${st.epochs} epochs = ${st.derived} optimizer steps`:'No audit yet';list('unknowns',data.audit.remaining_unknowns||[]);list('corrections',data.audit.implemented_corrections||[]);el('recipe').textContent=JSON.stringify(data.recipe,null,2);audit();table('run-table',['Workflow','Status','Priority / pool','Started (UTC)','Ended (UTC)'],[...data.runs].reverse().map(r=>{const a=node('a',r.name);a.href=r.url;a.target='_blank';a.rel='noreferrer';const b=badge(r.status||'not recorded',r.status?.startsWith('FAILED'));const n=node('div');n.append(b);if(r.failure_stage)n.append(node('small',r.failure_stage));return[a,n,`${r.priority||'—'} / ${r.pool||'—'}`,r.start_time,r.end_time];}));table('evidence',['Source record','SHA256'],data.evidence.map(e=>[e.path,e.sha256]));}
function promptDiagnostics(){const rows=(data.prompt_diagnostics||[]).flatMap(d=>Object.entries(d.arms).map(([arm,r])=>[arm,`${r.usable_candidates}/${r.candidates}`,`${r.clean_baseline_states}/${r.states}`,r.positive_selected_with_clean_baselines,r.admission?.['2048']?.selected_traces_with_clean_baselines,r.admission?.['4096']?.selected_traces_with_clean_baselines]));if(!rows.length){el('prompt-diagnostics').textContent='No completed prompt comparison imported yet.';return;}table('prompt-diagnostics',['Model / prompt','Usable reasoning','Valid-baseline states','Valid positive traces','Fit 2,048','Fit 4,096'],rows);}
function astraFollowup(){const v=data.status.astra_expert;if(!v)return;const gate=data.status.overnight?.astra_gate;el('astra-status').textContent=gate?`Full-tree solver test: ${num(gate.exploitability)} exploitability across ${num(gate.infosets)} information states; required ≤${gate.maximum}. ${gate.passed?'Gate passed.':'Gate failed; neither Astra-expert SFT arm was started.'}`:v.status+' · API preflight: '+v.valid_actions+'/'+v.preflight_states+' valid action policies, '+v.valid_summaries+'/'+v.preflight_states+' public summaries.';table('astra-arms',['Condition','Teacher information','Training'],v.arms.map(a=>[a.label,a.hint,gate?.passed===false?'Skipped: solver gate failed':a.training_status]));}
function overnightStudy(){const v=data.status.overnight;if(!v)return;el('overnight-status').textContent='Target: 5 p.m. Pacific, Oct 22 · Astra gate: full-tree exploitability ≤0.10 · '+(v.updated_utc||'');table('overnight-studies',['Study','State','Details'],(v.studies||[]).map(s=>[s.label,s.status,s.detail]));}
function resultsSummary(){
  const overnight=data.status.overnight||{},roles=overnight.roles||{};
  const finite=e=>Number.isFinite(e.summary?.single?.exploitability?.mean)&&Number.isFinite(e.summary?.oracle_bon?.exploitability?.mean);
  const unique=new Map();
  for(const e of data.evaluations||[])if(finite(e))unique.set(e.summary.checkpoint||e.label,e);
  const evaluations=[...unique.values()];
  const best=rows=>rows.reduce((a,b)=>!a||b.summary.oracle_bon.exploitability.mean<a.summary.oracle_bon.exploitability.mean?b:a,null);
  const score=(e,regime)=>e.summary[regime].exploitability;
  const scoreText=(e,regime)=>`${num(score(e,regime).mean)} ± ${num(score(e,regime).std)}`;
  const frontierRows=evaluations.filter(e=>e.label.startsWith('overnight-20261022-frontier/')&&e.label.includes('/frontier_4032_n1/eval/'));
  const frontier=best(frontierRows);
  const frontierRounds=evaluations.filter(e=>e.label.startsWith('overnight-20261022-frontier/')&&/\/leact\/round-\d+\/eval\//.test(e.label));
  const frontierRole=roles.frontier||{},stage=frontierRole.scheduler_status||frontierRole.stage;
  const epochStep=e=>Number(e.summary.checkpoint?.match(/checkpoint-(\d+)$/)?.[1]||0);
  const lastEpoch=rows=>rows.reduce((a,b)=>!a||epochStep(b)>epochStep(a)?b:a,null);
  const finalRound=frontierRounds.filter(e=>e.label.includes('/leact/round-2/eval/'));
  const finalEpoch=lastEpoch(finalRound);
  const bestSingle=frontierRounds.reduce((a,b)=>!a||score(b,'single').mean<score(a,'single').mean?b:a,null);
  const frontierComplete=stage==='COMPLETED'&&finalRound.length===3&&frontierRounds.length===6;
  const oldAnchor=best(evaluations.filter(e=>e.label.startsWith('reasoning-run-')&&e.label.endsWith('/eval/coldstart_thinking')));
  const oldLeact=best(evaluations.filter(e=>e.label.startsWith('reasoning-run-')&&e.label.includes('/leact/leact/round-1/eval/')));
  const oldExit=best(evaluations.filter(e=>e.label.includes('/exit/exit/round-3/eval/')));
  const originalBC=evaluations.find(e=>e.label==='Original BC · final epoch only');
  const correctedBC=best(evaluations.filter(e=>e.label.includes('/bc/eval/bc_epochs/')));
  const frontierTraining=(data.live_training||[]).find(t=>t.run==='overnight-20261022-frontier'&&t.phase.startsWith('frontier_4032_n1/'));
  const roundTraining=(data.live_training||[]).filter(t=>t.run==='overnight-20261022-frontier'&&t.phase.startsWith('leact/round-')).at(-1);
  const paragraphs=[],historical=[],rows=[];
  const paragraph=(lead,text)=>{const p=node('p');if(lead)p.append(node('strong',lead+' '));p.append(document.createTextNode(text));paragraphs.push(p);};
  const addScore=(label,e,note)=>{if(e)rows.push([label,scoreText(e,'single'),scoreText(e,'oracle_bon'),note]);};
  el('results-as-of').textContent=`Based on the snapshot imported ${new Date(data.storage.imported_utc).toLocaleString()}. Interpretation updates with the recorded results.`;
  el('results-headline').textContent=frontierComplete&&frontier
    ?`Both LeAct rounds are complete. Single-policy exploitability fell from ${num(score(frontier,'single').mean)} to ${num(score(finalEpoch,'single').mean)} at the final epoch.`:frontier
    ?frontierRounds.length?'Frontier warmstart and subsequent LeAct evaluations are available. The full reproduction still needs assessment.':'We have a promising frontier warmstart. The full LeAct result is still unproven.'
    :'The recorded experiments do not yet establish a successful paper reproduction.';
  paragraph('What the score means.','Exploitability measures how much a perfectly adapting opponent can take advantage of the policy. Lower is better; zero is ideal. It is not a win rate or a percentage of wrong answers. “Single” uses one sampled policy at each decision. “Oracle best-of-eight” lets the CFR solver choose the candidate closest to its strategy at each decision.');
  if(frontier){
    const dataset=frontierTraining?.dataset,steps=frontierTraining?.history.at(-1)?.step;
    paragraph('The frontier warmstart uses Astra explanations and CFR actions.',`Qwen3-8B learned from Astra-written public explanations paired with CFR solver action targets${dataset?` across ${num(dataset.infosets)} information states (${num(dataset.sft_examples)} forward/backward examples)`:''}${steps?`, finishing ${num(steps)} optimizer steps`:''}. Its selected evaluated checkpoint scores ${scoreText(frontier,'single')} on a single sampled policy and ${scoreText(frontier,'oracle_bon')} with oracle best-of-eight.`);
    if(frontierComplete){
      paragraph('The additional LeAct SFT improved the single-policy score.',`The final epoch scores ${scoreText(finalEpoch,'single')}, about ${(score(frontier,'single').mean/score(finalEpoch,'single').mean).toFixed(1)} times lower exploitability than the warmstart. All three epochs of both rounds were evaluated over all 4,032 states with inference seeds 0, 1 and 2. The best measured single-policy checkpoint scores ${scoreText(bestSingle,'single')}; that is a separate selection from the checkpoint retained using oracle best-of-eight.`);
      paragraph('The paper’s numerical target is reached in this reconstruction.',`Our final-epoch mean is below the paper’s LeAct reference of 0.016 ± 0.004. The warmstart corpus and prompts differ from the authors’ setup, the evaluation uses seen states, and the direct-action BC baseline remains unmatched. This supports the warmstart-plus-LeAct recipe without establishing an exact reproduction or a generalization advantage.`);
    }else paragraph('The very low best-of-eight score needs the solver.',`The model can generate candidates that collectively form a strong policy when the expert chooses among them. Its own first samples are less reliable. The warmstart single-policy score is ${(score(frontier,'single').mean/0.016).toFixed(1)} times the paper’s final LeAct mean of 0.016. A strong candidate pool alone does not establish that the model can choose well independently.`);
    const paperComparison=node('p');
    paperComparison.append(document.createTextNode('For context, the paper reports 1.351 ± 0.226 / 0.107 ± 0.026 for its warmstart and 0.016 ± 0.004 / 0.006 ± 0.002 for final LeAct (single / oracle best-of-eight). Our warmstart uses reconstructed prompts and an Astra corpus; the paper does not give the exact 6r2s reasoning-corpus size. These are reference targets, and this is not an identical-data reproduction. '));
    const link=node('a','Paper: Table 8 and Appendix H.3');link.href=data.paper;link.target='_blank';link.rel='noreferrer';paperComparison.append(link);historical.push(paperComparison);
    addScore('Frontier warmstart · best evaluated epoch',frontier,`${frontierRows.length} epoch evaluations available; warmstart only`);
  }
  if(frontierRounds.length){
    const rounds=[...new Set(frontierRounds.map(e=>Number(e.label.match(/\/leact\/round-(\d+)\/eval\//)[1])))].sort((a,b)=>a-b);
    for(const round of rounds){
      const epochs=frontierRounds.filter(e=>e.label.includes(`/leact/round-${round}/eval/`));
      const retained=best(epochs),measured=lastEpoch(epochs);
      paragraph(`Frontier LeAct round ${round}.`,`The latest evaluated epoch scores ${scoreText(measured,'single')} single and ${scoreText(measured,'oracle_bon')} with oracle best-of-eight. The checkpoint selected by lowest oracle best-of-eight scores ${scoreText(retained,'single')} single. These selection rules can choose different epochs.`);
      addScore(`Frontier LeAct · round ${round} · latest epoch`,measured,epochs.length===3?'Final epoch; all three epochs evaluated':'Latest available epoch');
    }
    if(frontierComplete)addScore('Frontier LeAct · best measured single-policy checkpoint',bestSingle,'Best single mean across six evaluated checkpoints');
  }else if(frontier){
    const selected=roundTraining?.dataset||frontierRole.progress?.dataset,last=roundTraining?.history.at(-1);
    const admission=selected?`The filter admitted ${num(selected.selected_traces)} traces across ${num(selected.states_with_selected_traces)} states, producing ${num(selected.sft_examples)} SFT examples. `:'';
    const recorded=last?`The last collected training log shows ${num(last.step)}/${num(last.total_steps)} steps${roundTraining.completed?'; a training completion record is present':'; no training completion record has been collected'}. `:'';
    const workflow=(data.runs||[]).find(r=>r.name===frontierRole.workflow);
    const canceledByQuota=workflow?.cancelled_by==='groot-idle-job-shutdown';
    const state=stage==='FAILED_CANCELED'?`${canceledByQuota?'The cluster quota enforcer':'OSMO'} canceled this workflow. No post-round evaluation was collected, so we cannot tell whether the added SFT improved poker play.`:`Workflow status: ${stage||'not recorded'}. No post-round evaluation is available yet.`;
    paragraph('The first LeAct round has no completed score.',admission+recorded+state);
  }
  if(oldAnchor&&oldLeact){
    paragraph('The earlier small warmstart gave mixed results.',`After one LeAct round, single-policy exploitability moved from ${num(score(oldAnchor,'single').mean)} to ${num(score(oldLeact,'single').mean)} (worse), while oracle best-of-eight moved from ${num(score(oldAnchor,'oracle_bon').mean)} to ${num(score(oldLeact,'oracle_bon').mean)} (better). That trained model’s candidate pool scored better, but its first sampled policy scored worse. The small pilot is a different recipe from the new frontier warmstart.`);
    historical.push(paragraphs.pop());
    addScore('Earlier small self-generated warmstart',oldAnchor,'Small pilot, different data coverage and learning rate');
    addScore('Earlier LeAct · round 1',oldLeact,'Single worsened; oracle best-of-eight improved vs its own warmstart');
  }
  if(oldExit){
    paragraph('The earlier ExIt baseline also remained short of reproduction.',`Its third round scored ${num(score(oldExit,'single').mean)} single and ${num(score(oldExit,'oracle_bon').mean)} with oracle best-of-eight. This is the forward-sampling baseline from the earlier warmstart; it is not a matched comparison with the new frontier run.`);
    historical.push(paragraphs.pop());
    addScore('Earlier ExIt · round 3',oldExit,'Earlier warmstart, not matched to the frontier run');
  }
  if(originalBC||correctedBC){
    const measured=correctedBC||originalBC;
    paragraph('The direct-action BC baseline is also not reproduced.',`BC trains the action targets without reasoning traces. The ${correctedBC?'corrected':'original'} BC checkpoint scores ${num(score(measured,'single').mean)} single and ${num(score(measured,'oracle_bon').mean)} with oracle best-of-eight, far from the paper’s 0.003 / 0.001. Our current comparisons therefore do not establish that reasoning is necessary or better than a correctly reproduced direct-action baseline.`);
    historical.push(paragraphs.pop());
    addScore('Original direct-action BC',originalBC,'Final epoch only; earlier loss-mask implementation');
    addScore('Corrected direct-action BC',correctedBC,'Best evaluated epoch; paper BC score not matched');
  }
  const selfRows=evaluations.filter(e=>e.label.startsWith('overnight-20261022-self/'));
  const projection=data.status.projection_sampling;
  if(projection){
    paragraph('New ablation: projection sampling before frontier warmstart SFT.',`The student rewrites the original frontier explanations using the sampling method from arXiv:2610.02140. The source corpus, solver action targets, base model and SFT budget stay matched to the completed control. A 32-state pilot checked policy preservation and token limits before all 4,032 states, warmstart SFT and two LeAct rounds. Status: ${projection.scheduler_status}; stage: ${(projection.progress?.phase||'H100 startup').replaceAll('_',' ')}. Original and rewritten traces, accepted proposals, fallbacks and evaluations are recorded.`);
    const projectionRows=evaluations.filter(e=>e.label.startsWith('projection-sampling-20261023-v3/'));
    const projectedWarm=lastEpoch(projectionRows.filter(e=>e.label.includes('/projection_4032_n1/eval/')));
    const projectedFinalEpochs=projectionRows.filter(e=>e.label.includes('/leact/round-2/eval/'));
    const projectedFinal=lastEpoch(projectedFinalEpochs);
    const projectionComplete=projection.scheduler_status==='COMPLETED'&&projectedFinalEpochs.length===3;
    if(projectedWarm){
      const controlWarm=lastEpoch(frontierRows);
      paragraph('Projection warmstart result.',`The latest evaluated warmstart epoch scores ${scoreText(projectedWarm,'single')} single-policy exploitability${controlWarm?`, compared with ${scoreText(controlWarm,'single')} for the original frontier warmstart`:''}. Lower is better. ${projectionComplete?'Both subsequent LeAct rounds are now complete.':'The two-round LeAct comparison still needs its final result.'}`);
      addScore('Projection warmstart · latest epoch',projectedWarm,'Matched corpus and budget; explanations rewritten');
    }
    if(projectionComplete&&finalEpoch)paragraph('Projection did not improve the final single-policy result.',`After two LeAct rounds, the projected arm scores ${scoreText(projectedFinal,'single')}, versus ${scoreText(finalEpoch,'single')} for the original frontier arm: ${(score(projectedFinal,'single').mean/score(finalEpoch,'single').mean).toFixed(1)} times higher exploitability. LeAct improves on the projected warmstart, but this rewrite adaptation remains behind the original frontier recipe. Higher likelihood alone did not produce better policy quality here; the pilot's changes in meaning prevent treating this as a clean test of information-preserving distribution alignment.`);
    for(const round of [1,2]){
      const epochs=projectionRows.filter(e=>e.label.includes(`/leact/round-${round}/eval/`));
      if(epochs.length)addScore(`Projection LeAct · round ${round} · latest epoch`,lastEpoch(epochs),epochs.length===3?'Final epoch; all three epochs evaluated':'Latest available epoch');
    }
    const pilot=data.status.projection_ablation?.first_pilot;
    if(pilot)paragraph('The initial projection pilot exposed a format problem.',`${pilot.rewrites_used}/${pilot.states} rewrites passed. Qwen placed its explanations outside the required reasoning block, and our parser included an end-of-message token in the policy JSON. SFT did not start.`);
    const secondPilot=data.status.projection_ablation?.second_pilot;
    if(secondPilot)paragraph('The second pilot also failed the training gate.',`${secondPilot.rewrites_used}/${secondPilot.states} rewrites passed despite ${secondPilot.accepted_proposals} accepted likelihood refinements. Outputs lost the required action structure or exceeded the length limit. The current revision rewrites only the explanation and copies the original solver action target outside the sampling search. It scores the complete explanation-plus-action response under the ordinary task prompt. Meaning preservation is requested but not formally verified.`);
    if(projection.retry)paragraph('H100 allocation retry.',`The quota enforcer canceled the corrected pilot before it started. Ordinary HIGH-priority retries use a backoff and preserve the two-node limit. ${projection.retry.waiting_for?.length?`The next attempt waits for ${projection.retry.waiting_for.join(', ')} to finish and release its H100 allocation.`:`The next attempt is scheduled for ${new Date(projection.retry.next_retry_epoch*1000).toLocaleTimeString()}.`}`);
    const measured=projection.sampling?.projection||projection.sampling?.pilot;
    if(measured)paragraph('Current projection sample checks.',`${measured.rewrites_used}/${measured.states} rewritten traces passed the format, solver-policy and context checks; ${measured.original_fallbacks} retained the original frontier trace. ${measured.accepted_proposals}/${measured.proposals} refinement proposals were accepted. These are sampler checks, not a poker evaluation.`);
    const thirdPilot=data.status.projection_ablation?.third_pilot;
    if(thirdPilot)paragraph('Passing the format gate does not establish preserved meaning.',`The revised pilot passed ${thirdPilot.rewrites_used}/${thirdPilot.states} format and target checks, but only ${thirdPilot.changed_explanations} explanations changed; ${thirdPilot.unchanged_explanations} remained identical. Qualitative review found ${thirdPilot.clear_semantic_drift_examples} clear examples of changed strategic advice or incorrect poker claims. The solver targets remain fixed. The SFT comparison therefore tests an approximate likelihood-rewrite adaptation; semantic equivalence has not been established.`);
  }
  const selfCells=['self_128_n1','self_4032_n1','self_4032_n4'].map(label=>lastEpoch(selfRows.filter(e=>e.label.includes(`/${label}/eval/`))));
  if(selfCells.every(Boolean)){
    paragraph('More self-generated traces produced a substantial scaling gain.',`The fixed final epochs score ${scoreText(selfCells[0],'single')} for 128 states × 1 trace, ${scoreText(selfCells[1],'single')} for 4,032 × 1, and ${scoreText(selfCells[2],'single')} when retaining up to four eligible traces per state. The last change lowers exploitability by ${(score(selfCells[1],'single').mean/score(selfCells[2],'single').mean).toFixed(1)} times at the same state coverage. These are warmstarts using base-model explanations with CFR action targets; no frontier explanations are used. Training stays at three epochs, so more traces also mean more optimizer steps and tokens. This is one training run per setting, evaluated with three inference seeds on seen states; it does not isolate data gains at fixed compute.`);
    selfCells.forEach((e,i)=>addScore(['Self warmstart · 128 × 1','Self warmstart · 4,032 × 1','Self warmstart · 4,032 × up to 4'][i],e,'Fixed final epoch; three epochs, increasing compute'));
  }else paragraph('The self-generated scaling comparison is incomplete.',selfRows.length?`${selfRows.length} completed checkpoint evaluation(s) have been imported. Compare the fixed-epoch cells for 128 states × 1 trace, 4,032 × 1 and 4,032 × 4 before claiming a scaling gain.`:`No completed scaling evaluation has been imported. Recorded workflow status: ${roles.self?.scheduler_status||roles.self?.stage||'not recorded'}. We do not yet know whether more self-generated traces help, or what the minimal successful recipe is.`);
  if(data.status.self_followup?.status==='waiting_for_projection')paragraph('Self-scaling remains queued.', 'The quota enforcer also reclaimed its latest node before training. Its validated recovery is queued after the projection run, so these studies can proceed one at a time when H100 capacity is available.');
  const selfLeact=data.status.self_leact;
  if(selfLeact){
    const measured=evaluations.filter(e=>e.label.startsWith('self-leact-20261024-v1/'));
    paragraph('LeAct after the self-generated warmstart.',`This separate arm starts from the fixed final epoch of the 4,032 × up-to-four-trace warmstart and runs two LeAct rounds with the frontier control's per-round settings. Workflow status: ${selfLeact.scheduler_status||'not recorded'}. The completed warmstart score does not include these rounds. Warmstart data and total compute are larger than in the frontier control, so this is not a causal comparison at equal compute.`);
    const selfFinalEpochs=measured.filter(e=>e.label.includes('/leact/round-2/eval/'));
    const selfFinal=lastEpoch(selfFinalEpochs);
    if(selfLeact.scheduler_status==='COMPLETED'&&selfFinalEpochs.length===3&&selfCells[2]){
      const reduction=100*(1-score(selfFinal,'single').mean/score(selfCells[2],'single').mean);
      paragraph('The self-generated arm shows a modest additional mean improvement.',`Single-policy exploitability changes from ${scoreText(selfCells[2],'single')} at warmstart to ${scoreText(selfFinal,'single')} after two LeAct rounds (${reduction.toFixed(1)}% lower mean). The change is small relative to the variation across the three inference seeds, and there is only one training run. This does not establish a reliable training gain. Oracle best-of-eight improves from ${scoreText(selfCells[2],'oracle_bon')} to ${scoreText(selfFinal,'oracle_bon')}, showing a stronger candidate pool when the CFR expert chooses the action policy; that is not autonomous single-policy performance.`);
    }
    for(const round of [1,2]){
      const epochs=measured.filter(e=>e.label.includes(`/leact/round-${round}/eval/`));
      if(epochs.length)addScore(`Self warmstart + LeAct · round ${round} · latest epoch`,lastEpoch(epochs),epochs.length===3?'Final epoch; all three epochs evaluated':'Latest available epoch');
    }
  }
  const shiftHypothesis=data.status.experiment_plan?.distribution_shift_hypothesis;
  if(shiftHypothesis){
    paragraph('Distribution-shift hypothesis.',shiftHypothesis.statement);
    paragraph('How to interpret the comparison.',shiftHypothesis.interpretation);
  }
  const gate=overnight.astra_gate;
  if(gate){
    paragraph('Astra as the action expert.',`Astra scored ${num(gate.exploitability)} across ${num(gate.infosets)} information states against your ≤${gate.maximum} requirement. ${gate.passed?'It passed that prerequisite; training outcomes still need separate evaluation.':'It failed, so the Astra-actions and Astra-three-task SFT experiments were skipped.'} This does not contradict the warmstart result: Astra supplied explanations there, while CFR supplied the correct action targets.`);
  }
  const workflows=[roles.frontier,roles.self].filter(v=>v?.workflow);
  if(workflows.length&&!(data.status.active_workflows||[]).length&&!workflows.some(v=>['RUNNING','PENDING','WAITING'].includes(v.scheduler_status)))
    paragraph('No GPU job is currently recorded as active.',`The frontier and self-scaling runs are ${workflows.map(v=>v.scheduler_status||v.stage).join(' / ')}. ${frontierComplete?'The frontier warmstart and both LeAct rounds have completed evaluations.':'Completed scores remain valid; an interrupted training curve is not a completed experiment.'}`);
  const builderbench=data.status.builderbench_preflight;
  if(builderbench)paragraph('BuilderBench simulator preparation.',`${builderbench.task_count} official tasks are indexed. ${builderbench.simulator_successes}/${builderbench.simulator_replays} diagnostic replays succeeded using scripted actions; those simulator checks made ${builderbench.expert_calls} expert-model calls. ${builderbench.passed?'The simulator preflight passed.':'The simulator preflight needs inspection.'} These checks establish environment operation, not GPT expert capability or a trained-model result.`);
  const builderbenchProbe=data.status.builderbench_probe;
  if(builderbenchProbe){
    const successes=builderbenchProbe.per_task.reduce((total,row)=>total+row.full_successes,0);
    paragraph('GPT-6 BuilderBench pilot completed.',`${successes}/${builderbenchProbe.rollout_count} rollouts fully succeeded; ${builderbenchProbe.full_task_pass_at_n_count}/${builderbenchProbe.task_count} tasks were solved at least once across two simulator seeds. Mean cube progress was ${num(builderbenchProbe.single_progress_mean)}, with ${builderbenchProbe.invalid_outputs} invalid outputs. This small, predeclared probe establishes feasibility, not full-benchmark expert quality. A fresh collection covers all 51 official tasks and retains failures as well as successes.`);
  }
  const expertProbe=data.status.builderbench_expert;
  const builderStudent=data.status.builderbench_student;
  if(builderStudent){
    const stages=Object.entries(builderStudent.roles||{}).map(([role,r])=>`${role}: ${r.scheduler_status}; ${(r.progress?.phase||'awaiting H100 startup').replaceAll('_',' ')}`);
    paragraph('BuilderBench student work is now automatic.',`${stages.join('. ')}. The frozen first corpus admits 153 action steps from 23 successful tasks: 153 action-only BC examples and 306 joint forward/backward examples. Eight over-context steps are excluded from both arms. The other 28 tasks remain outside training. A separate expert trajectory collection supplies LeAct and ExIt on the same 23 in-domain tasks. Final checkpoints are evaluated progressively toward 64 inference seeds; intermediate epochs use three. Missing or partial evaluations are not final benchmark scores.`);
    const expertIteration=data.status.builderbench_iteration_expert;
    if(expertIteration)paragraph('Separate demonstrations for iterative training.',`${expertIteration.completed_rollouts}/${expertIteration.planned_rollouts} rollouts completed, ${expertIteration.successful_rollouts} fully successful; status: ${expertIteration.status}. These trajectories do not enter the initial warmstart, and they do not expand the fixed training-task partition.`);
    const fixed=builderStudent.fixed_observation_speed;
    if(fixed)for(const comparison of fixed.comparisons||[])paragraph(`Speed on identical observations: ${comparison.student}.`,`${num(comparison.speedup)}× relative to ${comparison.reference} across ${comparison.states} frozen public observations. Action-match reward changes from ${num(comparison.reference_action_match_reward)} to ${num(comparison.student_action_match_reward)}; invalid responses: ${comparison.reference_invalid} → ${comparison.student_invalid}. This measures full generation on the same inputs, separately from complete task-rollout speed and success.`);
  }
  if(expertProbe){
    const probeRows=expertProbe.completed||[];
    const probeScope=expertProbe.study_kind==='teacher_collection'?`${expertProbe.planned_tasks} tasks, ${expertProbe.inference_seeds.length} inference seed(s) each. ${builderStudent?'Student workflow progress is described above.':'Student training has not started.'}`:'A fixed six-task, two-seed capability probe, not the full benchmark.';
    el('builderbench-status').textContent=`GPT-6 Astra · ${expertProbe.status} · ${expertProbe.completed_rollouts}/${expertProbe.planned_rollouts} rollouts completed; ${expertProbe.successful_rollouts} fully solved. ${expertProbe.task?`Current: ${expertProbe.task}, seed ${expertProbe.seed}, decision ${expertProbe.action_index+1}.`:''} ${probeScope} ${expertProbe.error||''}`;
    table('builderbench-rollouts',['Task','Seed','Cube progress ↑','Full success','Actions','Invalid outputs','End condition'],probeRows.map(r=>[r.task,r.seed,num(r.cube_progress),r.success?'Yes':'No',r.actions,r.invalid_outputs,r.stop_reason]));
    el('builderbench-examples').replaceChildren(...(data.status.builderbench_examples||[]).map(r=>{
      const detail=node('details');detail.append(node('summary',`${r.task} · ${r.seed} · decision ${Number(r.step)+1}`),node('p',r.decision_summary),node('pre',JSON.stringify(r.action,null,2)));
      const observations=node('details');observations.append(node('summary','Before and after'),node('pre',r.observation),node('pre',r.next_observation));detail.append(observations);return detail;
    }));
  }else{
    el('builderbench-status').textContent='GPT-6 expert selection is recorded in the experiment plan; no live probe has been imported.';
    el('builderbench-rollouts').replaceChildren();el('builderbench-examples').replaceChildren();
  }
  const speed=data.status.builderbench_speed;
  el('builderbench-speed-status').textContent=speed?`${speed.status}. Astra uses a remote API; students use the recorded GPU deployment. The base-student comparison measures changes under the same hardware and decoding settings.`:'Timing evidence has not been imported yet.';
  table('builderbench-speed-runs',['Run','Completed / planned','Full success','Cube progress','Mean decision (s)','Median / p95 (s)','Mean model time / rollout (s)','Mean total rollout (s)','Mean output tokens'],(speed?.reports||[]).map(r=>[
    r.label,`${r.completed_rollouts}/${r.planned_rollouts}`,`${r.full_successes}/${r.completed_rollouts}`,num(r.mean_cube_progress),num(r.decision_seconds.mean),`${num(r.decision_seconds.median)} / ${num(r.decision_seconds.p95)}`,num(r.rollout_model_seconds.mean),num(r.rollout_wall_seconds.mean),num(r.completion_tokens.mean)
  ]));
  table('builderbench-speed-comparisons',['Comparison','Matched rollouts','Model rollout speedup','Decision speedup','Total rollout speedup','Full success: reference → student','Speedup on both-successful rollouts'],(speed?.comparisons||[]).map(c=>[
    `${c.reference} → ${c.student}`,c.matched_rollouts,c.all_matched?`${num(c.all_matched.model_rollout_speedup)}×`:c.status,c.all_matched?`${num(c.all_matched.decision_speedup)}×`:'—',c.all_matched?`${num(c.all_matched.end_to_end_speedup)}×`:'—',c.all_matched?`${c.all_matched.reference_successes} → ${c.all_matched.student_successes}`:'—',c.both_successful?`${num(c.both_successful.model_rollout_speedup)}× (${c.both_successful.rollouts} pairs)`:'—'
  ]));
  el('results-explanation').replaceChildren(...paragraphs);
  el('results-history').replaceChildren(...historical);
  const plan=data.status.experiment_plan;
  if(plan){
    table('results-plan',['Order','Experiment','Plan / recorded status','Next milestone'],plan.sequence.map((step,index)=>[
      index+1,step.title,step.status,step.next_milestone
    ]));
    el('results-plan-note').textContent=plan.dashboard_note;
  }else{
    el('results-plan').replaceChildren();
    el('results-plan-note').textContent='No experiment plan has been imported.';
  }
  if(frontier){
    rows.push(['Paper warmstart','1.351 ± 0.226','0.107 ± 0.026','Table 8 reference']);
    rows.push(['Paper final LeAct','0.016 ± 0.004','0.006 ± 0.002','Table 8 target']);
    rows.push(['Paper direct-action BC','0.003 ± 0.001','0.001 ± 0.0002','Table 8 baseline']);
  }
  table('results-scorecard',['Experiment','Single exploitability ↓','Oracle best-of-eight ↓','Interpretation'],rows);
  el('results-reading-guide').replaceChildren(...[
    'Training loss is how well the model predicts the text in that run’s training examples. A falling curve confirms better fit to that dataset; it does not establish stronger poker play. Different datasets and training targets make losses across runs unsuitable as a leaderboard.',
    'The frontier warmstart trains on all 4,032 information states and is evaluated on that same full tree. It measures imitation on seen states, not generalization to unseen states or games. We have not isolated whether the explanations themselves cause the improvement over direct action training.',
    'The earlier 38-example pilot and the full frontier warmstart differ in data coverage, reasoning source and learning rate. Their gap cannot be attributed to the frontier model alone. Repeated historical checkpoint logs collected during retries are not independent training replicas.',
    'Reported ± values are standard deviations over three inference seeds for one trained checkpoint, not three separately trained models. A mean over inference seeds does not establish training-run reproducibility.'
  ].map(text=>node('p',text)));
  const card=el('cards').lastElementChild;
  if(stage==='FAILED_CANCELED'&&card){card.querySelector('.value').textContent='Interrupted';card.querySelector('.detail').textContent='Frontier workflow canceled; completed scores remain available';}
  else if(['PENDING','WAITING'].includes(stage)&&card){card.querySelector('.value').textContent='Queued';card.querySelector('.detail').textContent='Frontier continuation waiting for an H100 node';}
  else if(stage==='RUNNING'&&card){card.querySelector('.value').textContent='Running';card.querySelector('.detail').textContent=frontierRole.stage||'Frontier continuation';}
}
function embodiedStudy(){
  const study=data.status.experiment_plan?.embodied_first_exam?.dashboard;
  if(!study||!el('efe-summary'))return;
  el('efe-summary').textContent=study.summary;
  const current=node('span',`${study.phase} · Updated ${new Date(study.updated_utc).toLocaleString()}`);
  if(study.workflow){const a=node('a',` · ${study.workflow} ↗`);a.href=`https://us-west-2-aws.osmo.nvidia.com/workflows/${encodeURIComponent(study.workflow)}`;a.target='_blank';a.rel='noreferrer';current.append(a);}
  el('efe-current').replaceChildren(current);
  const counts=[['Train candidates',study.counts.train_candidates,'Eligibility before successful-trace admission'],['Development tasks',study.counts.development,'Recipe and checkpoint selection'],['Core evaluation',study.counts.core,'Frozen evaluation-only instances'],['Admitted teacher episodes',study.counts.admitted_teacher_episodes,`${study.counts.admitted_native_episodes} native episodes admitted`]];
  el('efe-counts').replaceChildren(...counts.map(([label,value,detail])=>{const c=node('div',undefined,'card');c.append(node('div',label,'label'),node('div',value,'value'),node('div',detail,'detail'));return c;}));
  const anyResult=study.arms.some(a=>Number.isFinite(a.result?.success));
  const trials=study.teacher_pilot?.trials||[],graded=trials.filter(r=>r.status==='graded');
  el('efe-result-note').textContent=anyResult?'Only measured results are shown; inspect coverage and protocol before comparing scores.':`${graded.length} Astra pilot/collection tasks graded, ${graded.filter(r=>r.score===1).length} successful. These are train/development tasks; core scores remain unmeasured. ${study.training_started?'Training has started.':'SFT has not started.'}`;
  table('efe-workers',['Teacher worker','Scheduler','Graded / planned','Successful','Current task','Last observed'],(study.workers||[]).map(w=>{const a=node('a',w.workflow||'Awaiting submission');if(w.workflow){a.href=`https://us-west-2-aws.osmo.nvidia.com/workflows/${encodeURIComponent(w.workflow)}`;a.target='_blank';a.rel='noreferrer';}return[a,w.scheduler_status||w.phase,`${w.graded}/${w.planned??'—'}`,w.successes,w.current_task||w.detail||'Pending',w.updated_utc?new Date(w.updated_utc).toLocaleString():'—'];}));
  el('efe-student-progress').textContent=study.student?.detail||'Base Qwen: checkpoint downloaded; serving and native evaluation integration in progress.';
  const seen=study.training_task_evaluation;
  const qualification=study.student_qualification;
  if(qualification?.plan?.model&&el('efe-qualification-summary')){
    const p=qualification.plan,s=qualification.status||{},a=qualification.published_audit||{};
    el('efe-qualification-summary').textContent=p.summary+' '+p.model_caveat+' '+(p.status||'');
    const model=node('a',p.model);model.href=p.model_url;model.target='_blank';model.rel='noopener';
    const baseline=node('a',p.baseline);baseline.href='#efe-direct-base-aligned';
    const state=node('a',s.status||p.status);
    if(s.workflow){state.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(s.workflow);state.target='_blank';state.rel='noopener';}
    const qualificationRows=[[model,s.graded?`${s.successes}/${s.graded}`:'Not measured',s.graded?s.control_steps:'Not measured',s.movement_verified==null?'Pending trace/position audit':String(s.movement_verified),baseline,state,s.detail||'Preparing the native baseline.']];
    for(const prior of qualification.attempts||[]){
      if(prior.excluded_from_task_success)continue;
      const h=prior.status||{},link=node('a','Previous serving condition');
      if(h.workflow){link.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(h.workflow);link.target='_blank';link.rel='noopener';}
      qualificationRows.push([p.model+' · v'+h.attempt,h.graded?`${h.successes}/${h.graded}`:'Not measured',h.control_steps??'Not measured','No issued control',baseline.cloneNode(true),link,(prior.serving_confounds||[]).join('; ')+'. Grades retained; corrected parser tested separately.']);
    }
    table('efe-qualification-status',['Student','Native successes / graded','Issued control steps','Movement verified','Nearest baseline','Status','Current activity'],qualificationRows);
    el('efe-qualification-plan').replaceChildren(...p.design.map(text=>node('p',text)));
    table('efe-qualification-gates',['Check','Requirement','State'],p.gates.map(g=>[g.name,g.criterion,g.status||'See native run status; no qualification result yet.']));
    el('efe-published-audit').textContent=a.summary||'Published Astra trace audit in progress; no action-only case has been confirmed.';
    table('efe-published-examples',['Published example','Native result','Interface evidence','Use in this study'],(a.examples||[]).map(e=>{const link=node('a',e.task);link.href=e.source_url;link.target='_blank';link.rel='noopener';return[link,e.outcome,e.finding,e.admission];}));
  }
  const diagnosis=study.trace_diagnosis;
  const scaled=study.pear_scaled;
  if(scaled?.plan?.arms&&el('efe-pear-scaled-summary')){
    const p=scaled.plan,s=scaled.status||{},e=p.environments||{},r=p.recipe||{};
    const proposal=p.proposal;
    const primaryUpdates=p.arms.reduce((total,a)=>total+(a.rl_updates??0),0);
    el('efe-pear-scaled-summary').textContent=proposal
      ?`PEAR++ launch preparation is authorized. Shared runtime: ${(s.phase||p.status).replaceAll('_',' ')}. ${e.task_variants??0} eligible EFE variants (${e.semantic_tasks??0} semantic tasks) across ${Object.keys(e.suite_counts||{}).length} candidate suites, plus the full native GR00T YAM catalog and eight generated reasoning families. These are catalog candidates; ${p.admitted_tasks??0} tasks are admitted to RL and ${primaryUpdates} primary updates are recorded across both runs. The target is 16 H100 nodes per arm, with a 34-node campaign limit; this is not an allocated GPU count.`
      :`Two PEAR++ arms are being prepared alongside LeAct. ${e.task_variants??0} EFE training variants across ${Object.keys(e.suite_counts||{}).length} candidate suites; ${p.admitted_tasks??0} tasks admitted to RL so far. Shared runtime: ${(s.phase||p.status).replaceAll('_',' ')}. No PEAR++ robot-success result is available yet.`;
    if(p.implementation?.current_summary)el('efe-pear-scaled-summary').textContent+=' '+p.implementation.current_summary;
    const modelChecks=scaled.model_checks?.arms||[];
    table('efe-pear-scaled-arms',['Run','Initialization / nearest baseline','Updates','Model engine check','What it tests'],p.arms.map(a=>{
      const baseline=node('a',a.warmstart_step?`MoE warmstart · step ${a.warmstart_step}`:'Base Qwen3.8-27B · evaluation pending');
      baseline.href=a.warmstart_step?'#efe-coverage-expansion-results':'https://huggingface.co/Qwen/Qwen3.8-27B';
      const check=modelChecks.find(c=>c.arm===(a.warmstart_step?'moe_warmstart496':'dense_base'));
      let checkCell='Awaiting runtime';
      if(check){checkCell=node('a',(check.worker?.phase||check.scheduler_status||'Pending').replaceAll('_',' ')+' · diagnostic');checkCell.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(check.workflow);checkCell.target='_blank';checkCell.rel='noopener';}
      return [a.model+' + PEAR++',baseline,`${a.rl_updates??0} · full fine-tuning`,checkCell,a.warmstart_step?'Can native-reward RL improve our existing warmstart across a broader task set?':'Can the paper’s dense base model learn directly on the same expanded task set?'];
    }));
    el('efe-pear-scaled-recipe').textContent=proposal
      ?'Authorized for both full-finetuning arms: 128 retained groups × 8 rollouts; AdamW LR 1e-6; PEAR score centering. Initial groups: native YAM 32, broader bimanual 24, general manipulation 24, language-conditioned control 16, and reasoning 32 (4 per family). Language following uses existing benchmark instructions, tasks and native success checks. Preserve group ratios after filtering and average the loss within each family before weighting. SGS selects learnable reset configurations within tasks; full-start evaluation stays fixed. Review at 32, 256 and 512 updates; continuation depends on per-task exposure and training/development success. Runtime and environment preparation jobs are submitted; primary RL is awaiting integration and qualification.'
      :`${r.retained_groups_per_update} retained groups × ${r.rollouts_per_group} rollouts per update; AdamW LR ${r.learning_rate}; score centering; one update using generated-token mean loss. LoRA disabled in both arms. The total update budget scales with admitted task identities; the final count is pending environment qualification. Constant-reward groups are discarded and still count toward compute cost.`;
    table('efe-pear-scaled-suites',['Candidate simulator suite','Training variants','Admission'],Object.entries(e.suite_counts||{}).map(([suite,count])=>[suite,count,'Action, native reward and scene checks pending']));
    const catalog=new Map(Object.entries(scaled.catalog_checks?.contracts||{}));
    if(catalog.size)table('efe-pear-scaled-suites',['Candidate simulator suite','Training variants','Admission'],Object.entries(e.suite_counts||{}).map(([suite,count])=>{
      const check=catalog.get(suite);
      return [suite,count,check?.success?'One native interface check passed; scene and policy checks pending':check?`Interface check needs repair: ${check.error}`:'Native interface check queued or running'];
    }));
    el('efe-pear-scaled-note').textContent='Existing simulator and reasoning benchmark holdouts stay protected; BEHAVIOR is excluded. Native GR00T YAM is separate from ABC-Sim and EFE YAMLab. Its environments are confirmed working by the study owner; enumerating the full catalog requires refreshing the expired GitLab login or an available checkout. These arms differ in architecture and initialization, so their comparison does not isolate warmstart alone.';
    if(proposal?.evaluation?.holdout_review){
      const h=proposal.evaluation.holdout_review;
      el('efe-pear-scaled-note').textContent+=` Frozen core membership includes ${h.core_cohort_counts?.same_sim_new_task??0} new-task candidates in proposed training simulators and ${h.core_cohort_counts?.same_sim_same_task_new_variant??0} same-task variant candidates whose scene separation still needs verification. Another ${h.unseen_simulator_reserve_count??0} entries are reserved across PandaDyn, RoboPaint, RoboTwin-2 and RoboWits. Final cohort labels require actual training-exposure and scene audits.`;
    }
    if(scaled.native_bridge?.worker?.success)el('efe-pear-scaled-note').textContent+=' Native EFE transport check passed: one action, fresh observation and official grade. This was a scripted API check, not a model success evaluation.';
    const names=scaled.checkpoint_names;
    if(names?.checkpoint_count)el('efe-pear-scaled-note').textContent+=` ${names.checkpoint_count} existing checkpoints have rp_jhh_ names; new committed checkpoints receive the same prefix.`;
    if(names?.checkpoints)table('efe-checkpoint-names',['Checkpoint name','Update','Contents','Cluster path'],Object.entries(names.checkpoints).map(([name,c])=>[node('code',name),c.step,c.kind.replaceAll('_',' '),node('code',c.path)]));
    if(s.workflow)el('efe-pear-scaled-workflow').href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(s.workflow);
  }
  const pear=study.pear_comparison;
  if(pear?.plan?.conditions&&el('efe-pear-summary')){
    const p=pear.plan,s=pear.status||{},q=pear.qualification||{},u=p.optimizer_update_qualification||{};
    el('efe-pear-summary').textContent='LeAct selects reasoning using expert-action likelihood; PEAR learns from native robot success rewards. Both will start from the same final warmstart. '+(u.result?.success?'The base-policy optimizer and checkpoint-reload check has passed.':u.workflow?'Base-policy optimizer check: '+pearCheckLabel(u).toLowerCase()+'.':'The optimizer check is in preparation.')+' No trained PEAR robot-success score is available yet.';
    table('efe-pear-conditions',['Condition','Nearest baseline','State','Native success','Interpretation'],p.conditions.map(c=>{
      const label=node('span',c.label);label.id='efe-pear-'+c.id;
      const baseline=node('a',({base:'Base Qwen',warmstart:'Warmstart','pear-sc':'PEAR with centering'})[c.baseline]||c.baseline);baseline.href=c.baseline==='base'?'#efe-coverage-expansion-results':'#efe-pear-'+c.baseline;
      const explanation=c.id==='pear-no-sc'?'Same RL recipe with score centering disabled; isolates its effect.':c.id==='pear-sc'?'Learn from native robot rewards with sampler-distribution drift correction.':c.id==='leact-r1'?'Select self-generated explanations using expert-action likelihood, then train.':'Common checkpoint for both post-training methods.';
      const stage=study.coverage_expansion?.training?.jobs?.find(j=>j.stage===c.id);
      const state=stage?.status?workflowLabel(stage.status):(c.id.startsWith('pear-')?(u.result?.success?'Optimizer check passed; waiting for warmstart':u.workflow?'Optimizer check: '+pearCheckLabel(u).toLowerCase():p.optimizer_qualification?'Preparing optimizer check':'Native rollout check'):c.status);
      return [label,baseline,state,'Pending matched evaluation',explanation];
    }));
    const r=p.rl_pilot||{};
    const native=pear.native_qualification||{};
    const nativeText=native.plan?.task?` Native pipeline check: eight base-student rollouts on training task ${native.plan.task}, with the original 96-call/1500-second episode and 32768-token response budgets. Every call retains timing, exact tokens and top-32 probabilities; failed and truncated outcomes remain recorded.`:'';
    el('efe-pear-recipe').textContent=`Primary comparison starts from the same verified final expanded warmstart and uses the frozen EFE split and constrained action interface. Planned pilot: ${r.retained_groups_per_update} usable groups × ${r.rollouts_per_group} native rollouts per update, up to ${r.optimizer_steps} updates; LR ${r.learning_rate}, token-mean REINFORCE, top-${r.sampling?.topk_log_probs} sampler probabilities. Zero-variance groups count toward rollout cost and have a bounded resampling budget. ${q.success?'Sampler/loss qualification passed. ':''}This is an EFE adaptation: model, LoRA, warmstart, context budget and scale differ from the paper. A real-update replay remains required before RL training.`+nativeText;
    if(u.workflow||s.workflow)el('efe-pear-workflow').href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(u.workflow||s.workflow);
    if(el('efe-pear-native-findings'))el('efe-pear-native-findings').textContent=(p.native_qualification_findings?.summary||'')+' '+(p.optimizer_qualification?.summary||'')+' '+(u.summary||'');
    if(el('efe-pear-qualification-groups'))table('efe-pear-qualification-groups',['Training task for optimizer validation','Verified native outcomes','Workflow','Purpose'],(p.optimizer_qualification?.groups||[]).map(g=>{
      const workflow=node('a',g.scheduler_status||'Prepared');
      if(g.workflow){workflow.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(g.workflow);workflow.target='_blank';workflow.rel='noopener';}
      return[g.task,g.native_rollouts?`${g.successes}/${g.native_rollouts} successes`:'Pending',workflow,'Base-policy qualification only; no primary RL updates.'];
    }));
  }
  const expansion=study.coverage_expansion;
  if(expansion?.plan?.candidate_tasks&&el('efe-coverage-expansion-summary')){
    const p=expansion.plan,s=expansion.status||{},training=expansion.training||{};
    const admitted=expansion.admission;
    const admittedSummary=admitted?.gate_passed?`Frozen training corpus: ${admitted.successful_episodes} successful episodes across ${admitted.unique_tasks} tasks, containing ${admitted.decisions} verified action decisions. Corpus admission is complete.`:'';
    const panelSize=Object.values(expansion.evaluation?.panel||{}).reduce((sum,n)=>sum+n,0);
    const baselines=['astra','base'].map(model=>(expansion.evaluation?.scores||[]).filter(r=>r.model===model).reduce((v,r)=>({graded:v.graded+r.graded,successes:v.successes+r.successes}),{graded:0,successes:0}));
    const baselineSummary=panelSize&&baselines.every(r=>r.graded===panelSize)?`The matched baseline panel is complete: Astra succeeded on ${baselines[0].successes}/${panelSize} tasks (${(100*baselines[0].successes/panelSize).toFixed(1)}%); base Qwen succeeded on ${baselines[1].successes}/${panelSize} (${(100*baselines[1].successes/panelSize).toFixed(1)}%). This is one rollout per task on our frozen screen, not the official full EFE score.`:'';
    const recovery=expansion.training_recovery||{};
    const intermediates=expansion.intermediate_evaluations?.runs||[],branches=expansion.branch_runs?.runs||[];
    const studentGraded=(expansion.evaluation?.scores||[]).filter(r=>!['astra','base'].includes(r.model)).reduce((n,r)=>n+r.graded,0);
    el('efe-coverage-expansion-summary').textContent=[expansion.execution_priorities?.primary_deliverable?`Priority: ${expansion.execution_priorities.primary_deliverable}. Target: ${expansion.execution_priorities.target_local}.`:null,baselineSummary,admittedSummary,
      recovery.last_recorded_update?`Expanded warmstart: ${recovery.last_recorded_update}/${recovery.planned_optimizer_updates} updates recorded; ${workflowLabel(recovery.status).toLowerCase()}. Latest saved checkpoint: ${recovery.committed_restart_step}. ${studentGraded?`${studentGraded} expanded-student rollouts graded; see the results below.`:'No expanded student robot-success result yet.'}`:s.summary||'Expanded teacher collection is queued.',
      ...intermediates.map(r=>{const graded=(expansion.evaluation?.scores||[]).filter(s=>s.model===r.id).reduce((n,s)=>n+s.graded,0);return `${r.label}: ${graded}/${r.planned_tasks} tasks graded; ${r.ready?'native evaluation released':(r.export?.phase||'preparing export').replaceAll('_',' ')}. This is an intermediate diagnostic; the full training run continues independently.`;}),
      ...branches.map(r=>{const job=training.jobs?.find(j=>j.stage===r.id);return `${r.label}: ${workflowLabel(job?.status)}. ${job?.progress?.detail||''} Uses the same training corpus and native evaluation panel as its parent.`;})].filter(Boolean).join(' ');
    table('efe-coverage-expansion-counts',['Eligible / candidate tasks','Native outcomes','Native successes','Candidate usable successes','Distinct successful tasks','Scene audit'],[[
      `${s.eligible_tasks??p.candidate_tasks}/${p.candidate_tasks}`,s.graded_episodes||0,s.native_successes||0,`${s.candidate_successful_episodes||0}/${p.target_successful_episodes}`,
      `${s.candidate_successful_tasks||0}/${p.minimum_distinct_successful_tasks} minimum`,s.scene_audit_complete?'Complete':'Pending']]);
    table('efe-coverage-expansion-jobs',['Teacher batch','Suite / scope','State','Graded','Infrastructure retries'],(s.jobs||[]).map(j=>{
      const a=node('a',j.id);if(j.workflow){a.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(j.workflow);a.target='_blank';a.rel='noopener';}
      return [a,j.suite||'Training candidates',j.status,`${j.graded}/${j.tasks}`,j.retry_count||0];
    }));
    const collectionResults=Object.entries(s.collection_by_suite||{}).map(([suite,r])=>`${suite}: ${r.successes}/${r.graded} completed teacher episodes succeeded`).join(' · ');
    const missing=s.missing_outcome_episodes?.length?`${s.missing_outcome_episodes.length} ended episode(s) have unavailable outcome evidence; excluded from these counts and not rerun as the same attempt. `:'';
    const deferred=(s.deferred_suites||[]).map(suite=>`${suite}: ${s.deferred_suite_reasons?.[suite]||'qualification pending'}`).join(' ');
    const allocations=Object.entries(s.allocation_policy?.maximum_concurrent_teacher_jobs_by_suite||{}).map(([suite,n])=>`${suite}: up to ${n} workers`).join(' · ');
    el('efe-coverage-expansion-recipe').textContent=`${collectionResults?`${collectionResults}. These are training-collection outcomes, not held-out evaluation scores. `:''}${missing}${allocations?`Future allocation limits: ${allocations}. `:''}${deferred?`Deferred from admission: ${deferred} `:''}Original candidate pool: ${Object.entries(p.suite_counts||{}).map(([k,v])=>`${k}: ${v}`).join(' · ')}. Coverage is a hypothesis, not an established sole cause. Repeated trajectories use the task's allowed frozen scene and do not count as new scene variants. After command and scene admission: fresh Qwen3.6 warmstart with three balanced pass-equivalents, then one LeAct round (8 candidates, up to 2 positive explanations, 50% warmstart replay). The optimizer budget scales with admitted states. ${training.summary||'Training waits for 256 verified successful episodes across at least 24 tasks.'} ${(training.jobs||[]).map(j=>`${j.stage}: ${j.progress?.detail||j.status}`).join(' · ')}`;
    if(el('efe-coverage-expansion-results')){
      const labels={astra:'Astra',base:'Base Qwen3.6',warmstart:'Expanded warmstart','leact-r1':'Expanded warmstart + LeAct'},baseline={base:'astra',warmstart:'base','leact-r1':'warmstart'};
      for(const r of intermediates){labels[r.id]=r.label+' (intermediate)';baseline[r.id]=r.baseline;}
      for(const r of branches){labels[r.id]=r.label;baseline[r.id]=r.baseline;}
      table('efe-coverage-expansion-results',['Condition','Evaluation cohort','Native success','Nearest baseline','Interpretation'],(expansion.evaluation?.scores||[]).map(r=>{
        const label=node('span',labels[r.model]||r.model);label.id=`efe-coverage-score-${r.model}-${r.cohort}`;
        const reference=baseline[r.model]?node('a',labels[baseline[r.model]]):node('span','Teacher reference');
        if(baseline[r.model])reference.href=`#efe-coverage-score-${baseline[r.model]}-${r.cohort}`;
        const cohort={seen_training_scene:'Seen training scenes',same_task_new_variant:'Same task, new variant',same_simulator_new_task:'Same simulator, new task'};
        return [label,cohort[r.cohort]||r.cohort,`${r.successes}/${r.graded} (${(r.success_rate*100).toFixed(1)}%)`,reference,
          r.cohort==='seen_training_scene'?'Behavior on admitted training scenes; not generalization.':r.cohort==='same_task_new_variant'?'New variant of a represented task family, with protected scene identity; one rollout per task.':'Frozen new tasks in represented simulators; one rollout per task.'];
      }));
    }
  }
  const fixed=study.fixed_action;
  const warm=fixed?.warmstart;
  if(warm?.plan?.spec&&el('efe-constrained-warm-summary')){
    const s=warm.status||{},h=warm.history||{},last=(h.history||[]).at(-1),spec=warm.plan.spec;
    const first=(h.history||[])[0];
    const fit=first&&last&&last.step>first.step?` Training reached ${last.step}/${spec.expected_optimizer_steps} updates: forward loss ${first.forward_ce.toFixed(2)} → ${last.forward_ce.toFixed(2)}, backward loss ${first.backward_ce.toFixed(2)} → ${last.backward_ce.toFixed(2)}. These measure fit to the teacher examples; robot success is measured separately.`:'';
    el('efe-constrained-warm-summary').textContent='Stronger-student warmstart: Qwen3.6-35B-A3B learns from 43 verified actions in three successful Astra episodes, with equal forward and backward explanation examples. This is a small language-LoRA pilot; broad EFE generalization is not yet measured.'+fit+' '+(warm.results?.summary||warm.plan.rerun_reason||'')+' '+(warm.results?.speed_note||s.progress?.detail||h.detail||'Preparing the H100 run.');
    const job=node('a','Qwen3.6 · warmstart LoRA');job.id='efe-qwen36-warm-lora';
    if(s.workflow){job.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(s.workflow);job.target='_blank';job.rel='noopener';}
    const baseline=node('a','Base Qwen3.6 · same constrained harness');baseline.href='#efe-fixed-qwen36-task';
    const loss=node('a','Interactive loss curves');loss.href='#loss-efe-warmstart';
    const runRows=[[job,`${last?.step||0}/${spec.expected_optimizer_steps}`,h.status||s.scheduler_status||'Prepared',baseline,loss,s.evaluation?.complete?`${s.evaluation.complete.successes}/${s.evaluation.complete.graded} succeeded`:s.evaluation?`${s.evaluation.status} · matched four tasks`:'Automatically follows the verified final checkpoint']];
    if(warm.leact?.plan?.spec){
      const l=warm.leact,ls=l.status||{},lh=l.history||{},ll=(lh.history||[]).at(-1);
      const run=node('a','Qwen3.6 · warmstart + LeAct R1');run.id='efe-qwen36-leact-lora';
      if(ls.workflow){run.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(ls.workflow);run.target='_blank';run.rel='noopener';}
      const parent=node('a','Warmstart · 1/4 tasks');parent.href='#efe-qwen36-warm-lora';
      const curves=node('a','Interactive loss curves');curves.href='#loss-efe-leact';
      const ev=ls.evaluation;
      runRows.push([run,`${ll?.step||0}/${l.plan.spec.expected_optimizer_steps}`,ls.progress?.detail||lh.status||ls.scheduler_status||'Prepared',parent,curves,ev?.complete?.success?`${ev.complete.successes}/${ev.complete.graded} succeeded`:ev?`${ev.status} · matched four tasks`:'Automatically follows verified LeAct export']);
      if(l.results?.summary)el('efe-constrained-warm-summary').textContent+=' '+l.results.summary;
      if(l.followup?.summary)el('efe-constrained-warm-summary').textContent+=' '+l.followup.summary;
      for(const j of l.followup?.jobs||[]){
        const link=node('a',j.kind==='diagnostic'?'43-observation action diagnostic':`${j.arm} · repeat seed ${j.seed}`);
        if(j.workflow){link.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(j.workflow);link.target='_blank';link.rel='noopener';}
        const reference=node('a',j.kind==='diagnostic'?'Warmstart versus LeAct':j.arm==='warmstart'?'Original warmstart':'Original LeAct');
        reference.href=j.arm==='leact'?'#efe-qwen36-leact-lora':'#efe-qwen36-warm-lora';
        const b=j.behavior||{},result=j.kind==='diagnostic'?'Action agreement and likelihood; no simulator score':`${b.successes||0}/${b.graded||0} graded · 3 planned training tasks`;
        runRows.push([link,'No weight updates',j.complete?.success?'Complete':j.progress?.detail||j.status,reference,'—',result]);
      }
      const support=(l.followup?.jobs||[]).find(j=>j.kind==='diagnostic')?.reasoning_comparison;
      if(support&&el('efe-leact-support')){
        const gain=g=>g?.mean_nats_per_command_token==null?'Pending':`${g.mean_nats_per_command_token.toFixed(3)} (${g.states} states)`;
        table('efe-leact-support',['Checkpoint','Teacher states measured','Valid responses','Exact teacher commands','Forward gain','Backward gain','Backward minus forward'],
          Object.entries(support).map(([arm,r])=>[arm,`${r.states}/${r.planned}`,`${r.valid_outputs}/${r.states}`,`${r.exact_commands}/${r.states}`,gain(r.forward_gain),gain(r.backward_gain),gain(r.backward_minus_forward)]));
        el('efe-leact-support-note').textContent='Diagnostic on fixed teacher observations, separate from robot success: greedy generation with a 4096-token cap. Likelihood gains are nats per executable-command token, averaged over matched states; positive means the explanation makes the teacher command more likely. Forward explanations are generated without the teacher action; backward explanations were generated with that action supplied. Exact agreement is not required for success because alternative commands may be valid.';
      }
    }
    table('efe-constrained-warm-status',['Run','Optimizer updates','State','Nearest baseline','Loss history','Robot evaluation'],runRows);
  }
  if(fixed?.plan?.summary&&el('efe-fixed-action-summary')){
    el('efe-completed-comparison').textContent=fixed.plan.results_summary||fixed.completed_comparison?.summary||'';
    el('efe-fixed-action-summary').textContent=fixed.plan.summary;
    el('efe-fixed-action-limits').textContent=fixed.plan.movement_gate+' '+fixed.plan.benchmark_separation+' '+fixed.plan.limitations;
    const fixedStates=new Map((fixed.status?.arms||[]).map(a=>[a.id,a]));
    table('efe-fixed-action-status',['Model / condition','Directed movements','Native task success','Control steps','Nearest baseline','State','Current activity'],(fixed.plan.arms||[]).map(planned=>{
      const a={...planned,...(fixedStates.get(planned.id)||{})};
      const link=node('a',(a.condition_label||(a.model==='astra'?'Astra':'Qwen3.6'))+' · '+a.mode);
      link.id='efe-fixed-'+a.id;
      if(a.workflow){link.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(a.workflow);link.target='_blank';link.rel='noopener';}
      const baseline=node('a',a.model==='qwen36'?'Astra · same interface':'Astra · original coding harness');
      baseline.href=a.model==='qwen36'?'#efe-fixed-astra-'+a.mode:'/efe-direct-control.html';
      return [link,a.mode==='movement'?(a.graded?`${a.movement_passes}/9`:'Not measured'):'Separate diagnostic',a.mode==='movement'?'Excluded: movement diagnostic':(a.graded?`${a.successes}/${a.graded}`:'Not measured'),a.graded?a.control_steps:'Not measured',baseline,a.status||'Prepared',a.detail||a.progress?.detail||'Queued in the experiment plan.'];
    }));
  }
  if(diagnosis?.totals){
    el('efe-trace-diagnosis-text').replaceChildren(...diagnosis.summary.map(text=>node('p',text)));
    const totals=diagnosis.totals;
    table('efe-target-audit',['Retained target type','States / 256','States admitted by LeAct','States drawn in LeAct','New LeAct example draws'],[['Error-returning tool targets','error_returning_target'],['Empty process polls','empty_poll_only'],['Image inspection only','image_only'],['Calls containing apply_patch','uses_apply_patch']].map(([label,key])=>[label,totals[key].decisions,totals[key].leact_selected,totals[key].leact_states_drawn,totals[key].leact_example_draws]));
    el('efe-trace-diagnosis-limits').textContent=diagnosis.limitations.join(' ');
  }
  const direct=study.direct_control;
  if(direct?.plan?.tasks){
    const p=direct.plan;
    el('efe-direct-control-plan').replaceChildren(...p.summary.map(text=>node('p',text)));
    table('efe-direct-control-tasks',['Training task','Control family','Phase','Existing successful demonstration'],p.tasks.map(t=>{
      const evidence=t.prior_teacher_episode?node('a','Watch source episode'):node('span','Not yet collected');
      if(t.prior_teacher_episode)evidence.href='/efe-rollouts.html#training/'+encodeURIComponent(t.prior_teacher_episode);
      return[t.task,t.control_family,t.pilot?'Six-task pilot':'Expansion after pilot gate',evidence];
    }));
    el('efe-direct-control-status').textContent=direct.status?.detail||'Prepared; waiting for an available authorized compute slot.';
    table('efe-direct-control-workers',['Collection','Scheduler','Graded','Successful','Eligible successful episodes','Current activity'],(direct.status?.workers||[]).map(w=>{
      const link=node('a',w.name);link.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(w.workflow);link.target='_blank';link.rel='noopener';
      return[link,w.scheduler,w.graded,w.successes,w.eligible_episodes,w.progress?.detail||'Awaiting startup'];
    }));
    table('efe-direct-control-students',['Checkpoint','Successes / graded','Coverage','Control steps','Nearest baseline','Status'],(direct.students||[]).map(run=>{
      const label=node('a',run.label);label.id='efe-direct-'+run.id;
      if(run.workflow){label.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(run.workflow);label.target='_blank';label.rel='noopener';}
      const baseline=node('a',run.id==='curated-leact'?'Curated warmstart · same tasks':run.id==='curated-warm64'?'Base Qwen · matched image processing':run.id==='warm96-direct'?'Base Qwen · same prompt':run.id==='base-aligned'?'Base Qwen · legacy image processing':'Astra · same prompt and tasks');
      baseline.href=run.id==='curated-leact'?'#efe-direct-curated-warm64':run.id==='curated-warm64'?'#efe-direct-base-aligned':(run.id==='warm96-direct'||run.id==='base-aligned')?'#efe-direct-base-direct':'/efe-direct-control.html';
      return[label,run.graded?`${run.successes}/${run.graded}`:'Not measured',`${run.graded||0}/${run.planned||6}`,run.graded?run.control_steps:'—',baseline,run.complete?.error||run.progress?.detail||run.status];
    }));
    el('efe-direct-control-preparation').textContent=direct.training_preparation?.detail||'The curated warmstart will use executed-action admission, verified task/scene isolation and measured context coverage before training.';
  }
  if(seen?.models){
    el('efe-training-task-scope').textContent=seen.scope;
    table('efe-training-task-scores',['Checkpoint','Success rate','Coverage','Matched starts','Control steps','Nearest baseline','Current activity'],seen.models.map(run=>{
      const name=node('span',run.label);name.id='efe-train-'+run.id;
      const baseline=node('a',run.id.startsWith('leact')?'Warmstart 96 on training tasks':'Original Astra demonstrations');
      baseline.href=run.id.startsWith('leact')?'#efe-train-warm96':'/efe-rollouts.html#training';
      return[name,run.graded?`${run.successes}/${run.graded} (${(100*run.success_rate).toFixed(1)}%)`:'Not measured',`${run.graded}/${run.planned}`,run.graded?`${run.matched_initial_states}/${run.graded}`:'Pending',run.graded?run.trials.reduce((a,t)=>a+t.control_steps,0):'—',baseline,run.complete?.error||run.progress?.detail||'Preparing checkpoint and simulator'];
    }));
    table('efe-training-task-suites',['Checkpoint','Suite','Success rate','Coverage','Episodes commanding control'],seen.models.flatMap(run=>(run.suites||[]).map(s=>[run.label,s.suite,s.graded?`${s.successes}/${s.graded} (${(100*s.successes/s.graded).toFixed(1)}%)`:'Not measured',`${s.graded}/${s.planned}`,s.graded?`${s.episodes_with_control}/${s.graded}`:'—'])));
    el('efe-training-task-interpretation').replaceChildren(...seen.models.map(run=>node('p',`${run.label}: ${run.behavior_note||'Waiting for measured behavior.'}`,'muted')));
    table('efe-training-task-behavior',['Checkpoint','Training task','Grader','Control steps','Simulator calls','Tool calls','Most repeated call'],seen.models.flatMap(run=>run.trials.map(t=>[run.label,t.task,t.score===1?'Pass':'Fail',t.control_steps,t.simulator_calls,t.tool_calls,t.most_repeated_call?`${t.most_repeated_call.count}× ${t.most_repeated_call.name}`:'—'])));
    el('efe-training-task-note').textContent=`${seen.serving_caveat} Updated ${new Date(seen.updated_utc).toLocaleString()}. Teacher source episodes were selected for successful training data; their 21/21 is not an unbiased teacher benchmark score.`;
  }
  table('efe-development',['Run','Successful / graded','Coverage','Simulator calls','Nearest baseline','Progress'],(study.development||[]).map(run=>{
    const name=node('span',run.name);name.id=`efe-dev-${run.id}`;
    const baseline=run.baseline?node('a',study.development.find(r=>r.id===run.baseline)?.name||run.baseline):node('span',run.id==='astra'?'Frontier reference':'Untouched model');
    if(run.baseline)baseline.href=`#efe-dev-${run.baseline}`;
    return[name,run.graded?`${run.successes}/${run.graded}`:'Not measured',`${run.graded}/${run.planned}`,run.graded?run.simulator_calls:'—',baseline,run.detail];
  }));
  const trainingProgress=node('span',study.training_pilot?.progress?.detail||'Warmstart preparation in progress.');
  if(study.training_pilot?.workflow){
    const link=node('a',` · ${study.training_pilot.workflow} ↗`);
    link.href=`https://us-west-2-aws.osmo.nvidia.com/workflows/${encodeURIComponent(study.training_pilot.workflow)}`;link.target='_blank';link.rel='noreferrer';trainingProgress.append(link);
  }
  el('efe-training-progress').replaceChildren(trainingProgress);
  table('efe-training-metrics',['Training run','Step','Forward / backward loss','LR','Processed / supervised tokens this step','Peak GPU memory','Seconds per step'],(study.training_runs||[]).filter(r=>r.history.length).map(run=>{
    const point=run.history.at(-1);
    return[run.name,`${point.step}/${point.total_steps}`,`${num(point.forward_ce)} / ${num(point.backward_ce)}`,num(point.learning_rate),`${num(point.processed_tokens)} / ${num(point.supervised_tokens)}`,`${num(point.gpu_memory_gib)} GiB`,num(point.seconds)];
  }));
  const group=el('efe-comparison').value;
  const selected=study.arms.filter(a=>group==='all'||a.group===group);
  const names=new Map(study.arms.map(a=>[a.id,a.name]));
  const t=node('table'),head=node('tr');['Run / planned arm','Core success','Measured coverage','Nearest baseline','What this comparison tests'].forEach(x=>head.append(node('th',x)));t.append(head);
  selected.forEach(arm=>{
    const tr=node('tr');tr.id=`efe-arm-${arm.id}`;tr.tabIndex=-1;
    const label=node('div');const dot=node('span',undefined,'dot');dot.style.backgroundColor=runColor(`efe/${arm.id}`);label.append(dot,document.createTextNode(' '+arm.name));
    const score=arm.result?.success,coverage=arm.result?.evaluated_core_tasks;
    const baseline=arm.baseline?node('a',names.get(arm.baseline)||arm.baseline):node('span','Reference');
    if(arm.baseline){baseline.href=`#efe-arm-${arm.baseline}`;baseline.addEventListener('click',event=>{event.preventDefault();el('efe-comparison').value='all';embodiedStudy();const target=el(`efe-arm-${arm.baseline}`);target?.scrollIntoView({block:'center',behavior:'smooth'});target?.focus({preventScroll:true});history.replaceState(null,'',baseline.href);});}
    [label,Number.isFinite(score)?`${(100*score).toFixed(1)}%`:'Not measured',Number.isInteger(coverage)?`${coverage}/${study.counts.core}`:'—',baseline,arm.explanation].forEach(value=>{const td=node('td');td.append(value instanceof Node?value:document.createTextNode(value));tr.append(td);});t.append(tr);
  });el('efe-arms').replaceChildren(t);
  el('efe-comparison-note').textContent=study.comparison_note;
  const retention=study.capability_retention;
  if(retention&&el('efe-capability-table')){
    el('efe-capability-description').textContent=retention.description+' '+(retention.interpretation||'');
    const stages=retention.stages;
    const metric=(stage,id)=>{
      const value=stage.result?.benchmarks?.[id];
      const box=node('div',value?`${(100*value.score).toFixed(1)}% (${value.correct}/${value.count})`:'Not measured');
      if(value)box.append(node('small',`${value.truncated} truncated · ${value.mean_output_tokens.toFixed(0)} output tokens on average`));
      const delta=stage.comparison?.[stage.base_id||'qwen-base']?.[id];
      if(delta){const sign=delta.delta_percentage_points>=0?'+':'';box.append(node('small',`${sign}${delta.delta_percentage_points.toFixed(1)} pp vs base · ${delta.correct_to_wrong} correct → wrong`));box.title=`Paired 95% interval: ${delta.paired_bootstrap_95_ci_pp.map(x=>x.toFixed(1)).join(' to ')} percentage points. ${delta.wrong_to_correct} wrong → correct.`;}
      return box;
    };
    table('efe-capability-table',['Checkpoint','Status / progress','HumanEval+ ↑','IFEval strict ↑','MMLU-Pro-280 ↑','Nearest baseline'],stages.map(stage=>{
      const name=node('span',stage.name);name.id=`efe-capability-${stage.id}`;
      const state=stage.progress?.detail||stage.scheduler?.status||stage.queue_note||(stage.status==='waiting_for_checkpoint'?'Waiting for checkpoint':'Prepared; awaiting submission');
      const baseline=stage.parent?node('a',stages.find(s=>s.id===stage.parent)?.name||stage.parent):node('span','Untouched model reference');
      if(stage.parent)baseline.href=`#efe-capability-${stage.parent}`;
      return[name,state,metric(stage,'humaneval_plus'),metric(stage,'ifeval'),metric(stage,'mmlu_pro_280'),baseline];
    }));
    el('efe-capability-note').textContent=retention.stage_note+' '+retention.comparison_note+' MMLU-Pro-280 is a fixed subset, not the official full benchmark score. These tests do not measure every aspect of language quality.';
  }
  table('efe-cohorts',['Evaluation group','Planned core cases','Interpretation'],study.cohorts.map(c=>[c.name,c.count,c.explanation]));
  el('efe-coverage-note').textContent=study.coverage_note;
  const recipe=study.recipe;
  table('efe-recipe',['Setting','Pilot specification'],[
    ['Model updates','Train language backbone and visual merger; freeze vision encoder'],
    ['Warmstart learning-rate candidates',recipe.warmstart_pilot.candidate_lrs.map(x=>x.toExponential()).join(' / ')],
    ['Training budget','Initially one balanced pass; development checkpoints at 0.25, 0.5 and 1 pass'],
    ['Context and generation',`${recipe.sequence_construction.training_context_tokens_initial.toLocaleString()} total training tokens; 8K / 16K / 32K completion-budget comparison; complete targets retained`],
    ['Data balance','Source mixture → suite → semantic task → episode → decision segment'],
    ['LeAct',`N=${recipe.leact_rounds.candidate_count}, top-${recipe.leact_rounds.max_positive_candidates} positive-delta traces; initial ${recipe.leact_rounds.initial_balanced_passes} pass, LR ${recipe.leact_rounds.initial_lr.toExponential()}, 50% warmstart replay`],
    ['Native demonstrations','Same admitted native actions, with versus without student backward-reasoning supervision; all holdouts protected']]);
  table('efe-gates',['Check','Status','Evidence / next step'],study.gates.map(g=>[g.id.replaceAll('_',' '),badge(g.status,g.status!=='passed'),g.detail]));
  table('efe-diagnostics',['Simulator control','Agent','Result','Interpretation'],(study.diagnostics.trials||[]).map(r=>[r.task,r.agent,r.status||'pending',r.detail||'Infrastructure diagnostic only']));
  el('efe-resources').textContent=study.resources;
}
el('efe-comparison')?.addEventListener('change',embodiedStudy);
let firstRender=true;
async function refresh(){try{const response=await fetch('/api/snapshot',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);data=await response.json();if(data.empty){el('status').textContent='No experiment data imported yet.';return;}render();promptDiagnostics();astraFollowup();overnightStudy();resultsSummary();LeActLeaderboards.render(data);embodiedStudy();currentStudyStatus();if(firstRender){firstRender=false;lossRoute();}}catch(e){el('updated').textContent='Refresh failed: '+e.message+' · Showing the last received snapshot.';}}
document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',event=>{activateView(b.dataset.tab);if(b.dataset.tab==='losses'){history.replaceState(null,'','#loss-'+chartState.category);if(data)chart();}else if(event.isTrusted){history.replaceState(null,'','#'+b.dataset.tab);}}));
for(const id of ['metric','chart-range','chart-smoothing'])el(id).addEventListener('change',chart);
el('loss-study-select').addEventListener('change',event=>{location.hash='loss-'+event.target.value;});
el('audit-filter').addEventListener('change',audit);
// Keep current EFE evidence first; older studies remain intact and deep-linkable.
const efePanel=el('leaderboard-efe'),efeHeading=efePanel.querySelector('.panel-head');
const expansionPanel=el('efe-coverage-expansion'),pearPanel=el('efe-pear-comparison');
efeHeading.querySelector('.eyebrow').textContent='Astra to Qwen3.6 35B';
const collectionDetails=node('details',undefined,'supporting-details');collectionDetails.append(node('summary','Collection and training details'));
for(const id of ['efe-coverage-expansion-counts','efe-coverage-expansion-jobs','efe-coverage-expansion-recipe'])collectionDetails.append(el(id));
expansionPanel.append(collectionDetails);
const qualificationDetails=node('details',undefined,'supporting-details');qualificationDetails.append(node('summary','Qualification results and training recipe'));
for(const id of ['efe-pear-recipe','efe-pear-native-findings','efe-pear-qualification-groups'])qualificationDetails.append(el(id));
pearPanel.append(qualificationDetails);
const earlier=node('details',undefined,'earlier-efe');earlier.append(node('summary','Earlier EFE pilots and original protocol'));
const oldComparison=efeHeading.querySelector('label');if(oldComparison)earlier.append(oldComparison);
for(const child of [...efePanel.children])if(![efeHeading,expansionPanel,pearPanel].includes(child))earlier.append(child);
efeHeading.after(expansionPanel,pearPanel);efePanel.append(earlier);
for(const detail of earlier.querySelectorAll('details'))detail.open=false;
el('benchmarks').insertBefore(efePanel,el('leaderboard-leduc'));
LeActLeaderboards.init();window.addEventListener('hashchange',lossRoute);
document.addEventListener('click',event=>{const a=event.target.closest('a[href^="#"]');if(a&&a.hash===location.hash)lossRoute();});
if(!location.hash)history.replaceState(null,'','#loss-efe-expanded');
lossRoute();refresh();setInterval(refresh,15000);
let chartResizeTimer;
window.addEventListener('resize',()=>{clearTimeout(chartResizeTimer);chartResizeTimer=setTimeout(()=>{if(data&&!data.empty)chart();},120);});
