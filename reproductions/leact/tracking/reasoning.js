'use strict';
function pearEstimate(p){
  if(Object.values(p).some(v=>!Number.isFinite(v))||p.updates<=0||(p.groups??128)<=0||(p.rollouts??8)<=0||(p.control_fraction??.75)<0||(p.control_fraction??.75)>1||p.retention<=0||p.retention>1||p.availability<=0||p.availability>1)throw Error('Enter finite positive inputs; fractions must be between zero and one.');
  if(p.rollout_gpus<4||p.rollout_gpus%4||p.trainer_gpus<8||p.trainer_gpus%8||p.simulator_gpus<0||Math.min(p.control_tokens,p.reasoning_tokens,p.tokens_per_node_second,p.parallel_environments,p.episode_seconds,p.learner_seconds_per_update)<=0||p.setup_hours<0)throw Error('Invalid timing, token or GPU layout assumption.');
  const fraction=p.control_fraction??.75,retained=p.updates*(p.groups??128)*(p.rollouts??8),control=retained*fraction/p.retention,reasoning=retained*(1-fraction)/p.retention;
  const generated=control*p.control_tokens+reasoning*p.reasoning_tokens;
  const sessions=Math.min(p.parallel_environments,p.rollout_gpus/4*16);
  const phases={'generation':generated/(p.tokens_per_node_second*p.rollout_gpus/8)/3600,'environment occupancy':control*p.episode_seconds/sessions/3600,'learner':p.updates*p.learner_seconds_per_update*8/p.trainer_gpus/3600};
  const wall=p.setup_hours+Math.max(...Object.values(phases))/p.availability,gpus=p.rollout_gpus+p.trainer_gpus+p.simulator_gpus;
  return {wall_hours:wall,allocated_gpu_hours:gpus*wall,gpus_per_run:gpus,retained_episodes:retained,attempted_episodes:control+reasoning,generated_tokens:generated,phase_hours:phases,bottleneck:Object.keys(phases).reduce((a,b)=>phases[a]>phases[b]?a:b)};
}
if(typeof module!=='undefined')module.exports={pearEstimate};
if(typeof document!=='undefined'){
  const $=id=>document.getElementById(id),fmt=n=>typeof n==='number'?n.toLocaleString(undefined,{maximumFractionDigits:1}):String(n);
  let data=null,page=0,filtered=[];const PAGE=40;
  function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
  function duration(h){return h>=48?fmt(h/24)+' days':fmt(h)+' hours';}
  function table(target,heads,rows){const t=node('table'),thead=node('thead'),hr=node('tr');heads.forEach(h=>hr.append(node('th',h)));thead.append(hr);t.append(thead);const body=node('tbody');rows.forEach(row=>{const tr=node('tr');row.forEach(value=>{const td=node('td');td.append(value instanceof Node?value:document.createTextNode(value??'—'));tr.append(td);});body.append(tr);});t.append(body);$(target).replaceChildren(t);}
  function textCell(title,detail){const n=node('div',title);if(detail)n.append(node('small',detail));return n;}
  function embodiedReasoning(){
    const r=data.embodied_reasoning||{},families=r.families||[],sources=r.existing_training_sources||[],admitted=families.reduce((n,f)=>n+(f.admitted_tasks||0),0),covered=new Set(sources.flatMap(s=>s.families));
    const trainScenes=families.reduce((n,f)=>n+(f.qualified_train_scenes||0),0),devScenes=families.reduce((n,f)=>n+(f.qualified_development_scenes||0),0);
    $('reasoning-status').textContent=`${families.length} required families · ${admitted} qualified task templates · ${trainScenes} train / ${devScenes} development scenes`;
    table('reasoning-task-table',['Training family','Tasks and output','Evaluation targets'],families.map(f=>{
      const name=textCell(f.name,f.transport_passed?`${f.qualified_train_scenes} train / ${f.qualified_development_scenes} development scenes; reward controls and PEAR transport passed`:covered.has(f.id)?'Existing training source identified; integration pending':f.status==='planned_adapter_and_generator_not_implemented'?'Generator or additional training source needed':f.status);
      const tasks=node('div'),example=node('p',f.example,'reasoning-example');tasks.append(node('div',f.output),example);
      const details=node('details'),summary=node('summary','Task list and scene sources'),list=node('ul');f.tasks.forEach(t=>list.append(node('li',t)));details.append(summary,list,node('p',f.training_environment));tasks.append(details);
      return [name,tasks,(f.transfer_targets||[]).join(', ')];
    }));
    table('reasoning-source-table',['Training dataset','Tasks / labels','Preparation status'],sources.map(s=>{const link=node('a',s.name);link.href=s.url;link.target='_blank';link.rel='noreferrer';const source=node('div');source.append(link,node('small',s.partition));return [source,s.skills,s.status];}));
    $('reasoning-benchmarks').replaceChildren(...(r.evaluation_benchmarks||[]).map(name=>node('span',name,'tag eval')));
    $('reasoning-bbox-status').textContent=trainScenes?'The procedural object-pointing family includes bounding-box answers checked by intersection over union. These are initial primitive-scene templates; the external datasets above are not yet integrated. Named benchmark datasets remain evaluation-only.':(r.box_output_planned?'Bounding-box outputs are specified in the proposal.':'Pointing uses normalized image coordinates.')+' Data and reward qualification are pending.';
  }
  function taskGroup(title,tasks,suite,kind){
    const group=node('section',undefined,'task-group '+kind);group.append(node('h4',title+' · '+fmt(tasks.length)));
    const list=node('ul');list.tabIndex=0;list.setAttribute('aria-label',title+' tasks');
    for(const task of tasks){const label=task.id.replace(suite+'-','').replace(/-(standard-realtime|standard|privileged)$/,'').replace(/-i(\d+)$/,' · variant $1').replaceAll('_',' ').replaceAll('-',' '),item=node('li',label);item.title=task.id;item.dataset.taskId=task.id;list.append(item);}
    group.append(list);return group;
  }
  function simulatorFamilies(){
    if(!data)return;
    const all=data.simulators?.families||[],query=$('sim-search').value.trim().toLowerCase(),split=$('sim-split').value;
    const families=all.filter(f=>(split==='all'||f[split]?.length)&&(!query||[f.name,f.id,...['train','eval_seen','eval_unseen','pending'].flatMap(k=>(f[k]||[]).map(t=>t.id+' '+(t.description||'')))].join(' ').toLowerCase().includes(query)));
    const counts=data.simulators?.counts||{};
    $('mixture-summary').textContent=`${fmt(counts.train||0)} train candidates · ${fmt(counts.eval_seen||0)} seen-sim / unseen-task eval tasks · ${fmt(counts.eval_unseen||0)} unseen-sim / unseen-task eval tasks`;
    $('simulator-count').textContent=fmt(families.length)+' families';
    const yam=data.native_yam_split||{},countsYam=yam.counts||{},groupsYam=yam.semantic_groups||{};
    $('yam-split-note').textContent=yam.entries?`Native GR00T YAM: ${countsYam.train} train names (${groupsYam.train} task groups), ${countsYam.development} development names and ${countsYam.test} test names (${groupsYam.test} task groups). All saved layouts remain evaluation-only; training requires independent scenes. Native GR00T YAM is separate from EFE YAMLab. Scene and exposure audits remain pending.`:'Native GR00T YAM membership has not been imported yet.';
    const cards=families.map(f=>{
      const card=node('article',undefined,'family-card');
      if(f.image){const figure=node('figure'),img=node('img');img.src=f.image.url;img.alt=f.name+' simulator preview';img.loading='lazy';img.decoding='async';figure.append(img);const caption=node('figcaption'),source=node('a',f.image.caption);source.href=f.image.source_url;source.target='_blank';source.rel='noreferrer';caption.append(source);figure.append(caption);card.append(figure);}
      const body=node('div',undefined,'family-body'),heading=node('div',undefined,'family-heading');heading.append(node('h3',f.name),node('span',f.split_status,'tag '+(f.eval_unseen.length?'eval':f.pending.length?'pending':'train')));body.append(heading);
      if(!f.pending.length)body.append(taskGroup('Train',f.train,f.id,'training'));
      if(f.eval_seen.length)body.append(taskGroup('Eval · seen sim, unseen task',f.eval_seen,f.id,'evaluation'));
      if(f.development?.length)body.append(taskGroup('Development · held-out tasks',f.development,f.id,'development'));
      if(f.eval_unseen.length)body.append(taskGroup('Eval · unseen sim, unseen task',f.eval_unseen,f.id,'evaluation'));
      if(f.pending.length)body.append(taskGroup('Catalog · train/eval split pending',f.pending,f.id,'pending'));
      if(!f.eval_seen.length&&!f.eval_unseen.length&&!f.pending.length)body.append(node('p','No eval tasks in these two categories.','meta'));
      card.append(body);return card;
    });
    $('simulator-families').replaceChildren(...cards);if(!cards.length)$('simulator-families').append(node('p','No matching simulator families.','meta'));
  }
  function liveExperiments(){
    const r=data.warmstart_retention||{},p=r.progress||{},questionCount=r.result?.questions??p.total??r.questions_per_model??765,preliminary=r.result?.preliminary??p.preliminary??r.preliminary??questionCount<765;
    $('retention-status').replaceChildren();
    const text=r.scheduler_status?`Warmstart-496 ${preliminary?'preliminary screen':'retention'} · RealWorldQA · ${r.scheduler_status}${p.model?' · '+p.model:''}${p.completed!==undefined?' · '+p.completed+'/'+questionCount+' questions':''}${p.phase?' · '+p.phase.replaceAll('_',' '):''}`:'Warmstart-496 retention · RealWorldQA · waiting for submission';
    const a=node(r.url?'a':'span',text);if(r.url){a.href=r.url;a.target='_blank';a.rel='noreferrer';}$('retention-status').append(a);
    const result=r.result;if(result)table('retention-results',['Checkpoint','Accuracy','Correct','Mean output tokens','Format failures','Truncations'],['base','warmstart496'].map(k=>[k,fmt(result[k].accuracy*100)+'%',Math.round(result[k].accuracy*questionCount)+'/'+questionCount,fmt(result[k].mean_output_tokens),result[k].format_failures,result[k].truncated]));
    $('retention-explanation').textContent=result?`Warmstart minus base: ${fmt(result.warm_minus_base*100)} percentage points; paired 95% interval [${result.paired_95ci.map(n=>fmt(n*100)).join(', ')}]. ${result.warm_minus_base<0?Math.round(-result.warm_minus_base*questionCount)+' fewer correct answers after warmstart. ':''}${result.paired_95ci[0]<=0&&result.paired_95ci[1]>=0?'The small sample does not establish a reliable difference; a meaningful regression remains possible. ':''}${preliminary?'Preliminary screen on '+questionCount+' seeded random questions. ':'Full visual-QA test split. '}Mean generated output shrank from ${fmt(result.base.mean_output_tokens)} to ${fmt(result.warmstart496.mean_output_tokens)} tokens; this is a token-count comparison. Coding and other language capabilities remain unmeasured.`:`${preliminary?'Preliminary screen: '+questionCount+' seeded random questions from the 765-question test split.':questionCount+' questions in the full test split.'} Base and warmstart-496 use the identical questions and reasoning budget. The comparison will report accuracy, a paired confidence interval, formatting failures and truncation. No paired result yet.`;
    const live=data.live_pear||{},runs=live.runs||{},entries=Object.entries(runs),latest=entries.map(([,v])=>v.heartbeat?.updated_utc).filter(Boolean).sort().at(-1);
    const collector=data.hosted_collector;
    $('live-status').textContent=collector?`Hosted collector · ${collector.success?'online':'needs attention'} · ${new Date(collector.updated_utc).toLocaleTimeString()}`:r.observed_utc?'Evaluation checked '+new Date(r.observed_utc).toLocaleTimeString():'';
    const launchChecks=entries.flatMap(([,v])=>Object.values(v.metadata?.launch_checks||{}));
    $('pear-live-status').textContent=latest?'PEAR heartbeat '+new Date(latest).toLocaleString()+' · '+(Date.now()-Date.parse(latest)>120000?'stale; check collector or workflow':'receiving metrics'):launchChecks.length?'Warmstart launch checks are being tracked. No primary RL update is recorded yet.':'No primary RL update is recorded yet. Training charts populate when a run reports metrics.';
    const arms=data.pear.arms||[];
    const pre=data.pear.preflight||{},controls=pre.controls?.suites||{},reasoningChecks=pre.reasoning?.worker?.families||{},yam=pre.native_yam?.worker||{},onpolicy=pre.on_policy||{};
    const controlRows=Object.entries(controls).map(([suite,value])=>{const liveCheck=launchChecks.find(c=>c.phase?.includes(suite)),worker=value.worker||{};return [suite,`${worker.checked??liveCheck?.checked??0} / ${worker.total??liveCheck?.total??'?'}`,worker.passed??liveCheck?.passed??0,value.scheduler_status??liveCheck?.scheduler_status??'pending'];});
    table('launch-validation',['Component','Checks completed','Checks passed','Status'],[
      ['Frozen MoE router + numerical restart','8 GPU ranks',pre.router?.worker?.result?.engine_check_passed?'passed':'pending',pre.router?.scheduler_status||'pending'],
      ['Procedural reasoning + PEAR transport',`${Object.keys(reasoningChecks).length} / 8 families`,Object.values(reasoningChecks).filter(c=>c.passed).length,pre.reasoning?.scheduler_status||'pending'],
      ...controlRows,
      ['Recorded control RGB observations',pre.camera_observations?.checked??0,pre.camera_observations?.passed??0,'reset and post-action images; partial audit'],
      ['Native GR00T YAM API',`${yam.checked||0} / ${yam.total||35}`,yam.passed||0,pre.native_yam?.phase?.replaceAll('_',' ')||'pending'],
      ['On-policy score-centering update','discarded qualification',onpolicy.worker?.passed?'passed':'pending',onpolicy.scheduler_status||'pending']]);
    $('launch-validation-explanation').textContent=`${pre.protected_scene_collisions||0} proposed control resets overlap reserved scene identities and are excluded from training. Passing scripted controls alone is insufficient: actual camera frames must also pass. The current RGB audit has ${(pre.camera_observations?.checked||0)-(pre.camera_observations?.passed||0)} failed tasks. The procedural bank has 512 train and 64 development scenes from eight initial task templates. Isaac camera workers require RTX hardware. Native YAM still needs a compatible runtime and independent training-reset qualification. The 16-H100 score-centering check uses a smaller discarded batch; its updates are excluded from primary training history.`;
    table('pear-run-table',['Run','Initialization / baseline','Updates observed','Explanation'],arms.map(arm=>{const run=runs[arm.id],step=run?.optimizer_steps_observed||0,active=run?.heartbeat?.updated_utc&&Date.now()-Date.parse(run.heartbeat.updated_utc)<=120000,checks=Object.values(run?.metadata?.launch_checks||{}),enabled=data.pear.enabled_training_arms?.includes(arm.id),router=data.pear.preflight?.router?.worker?.result,bridge=data.pear.preflight?.native_bridge?.worker;return [arm.id,arm.warmstart_step?'MoE · warmstart step '+arm.warmstart_step:'Qwen3.8-27B · base',step,step?'Training metrics below show observed updates; held-out evaluation is reported separately.':enabled?[router?.engine_check_passed?'Frozen-router update and exact reload passed.':'Router qualification pending.',bridge?.success?'Native reset/action/reward bridge already passed.':'Native bridge qualification pending.','RL execution checks remain; no primary RL update yet.'].join(' '):data.pear.paused_training_arms?.includes(arm.id)?'Paused by request; no primary optimizer update.':checks.length?checks.map(c=>c.phase.replaceAll('_',' ')+' · '+c.scheduler_status).join('; '):active?'Run is reporting; the first optimizer update is pending.':'On hold; no primary optimizer update or evaluation result yet.'];}));
    table('pear-recipe',['Run','Recipe status','LR','Group batch','Rollouts / group','Score centering','Parameter update','Router policy'],arms.map(arm=>{const m=runs[arm.id]?.metadata;return [arm.id,m?.run_id&&!m?.qualification_only?'Recorded configuration':'Planned configuration',m?.learning_rate??1e-6,m?.retained_groups??128,m?.rollouts_per_group??8,m?.score_centering===false?'off':'on',m?.parameter_update??'full finetuning',(m?.router_policy??arm.router_policy??'not applicable').replaceAll('_',' ')];}));
    table('pear-checkpoints',['Run','Completed checkpoint','Step','Recorded'],entries.flatMap(([key,v])=>(v.checkpoints||[]).map(c=>[key,c.name,c.step,new Date(c.updated_utc).toLocaleString()])));
    table('pear-evaluations',['Run','Evaluation step','Metric','Value'],entries.flatMap(([key,v])=>(v.evaluations||[]).flatMap(e=>Object.entries(e.metrics).map(([metric,n])=>[key,e.step,metric,fmt(n)]))));
    const select=$('metric-run'),old=select.value;while(select.options.length>1)select.remove(1);entries.forEach(([key])=>select.append(new Option(key,key)));select.value=[...select.options].some(o=>o.value===old)?old:'all';
    const metric=$('metric-name'),previous=metric.value,names=new Set(['policy_loss','grad_norm','learning_rate','entropy','rollout_policy_kl','response_tokens','truncation_rate','step_seconds','gpu_memory_gb','reward_unfiltered_cumulative/all','reward_unfiltered_cumulative/macro_task']);
    entries.forEach(([,v])=>v.history.forEach(row=>Object.keys(row.metrics).forEach(k=>names.add(k))));metric.replaceChildren(...[...names].map(k=>new Option(k.replaceAll('_',' '),k)));metric.value=names.has(previous)?previous:'policy_loss';
    table('live-task-rewards',['Run','Task','Cumulative raw success rate'],entries.flatMap(([key,v])=>Object.entries(v.latest_task_rewards).map(([task,value])=>[key,task,fmt(value*100)+'%'])));
    table('live-metrics',['Run','Metric','Latest value'],entries.flatMap(([key,v])=>Object.entries({...v.latest_metrics,...Object.fromEntries(Object.entries(v.rollout_diagnostics||{}).map(([k,n])=>['rollout_diagnostics/'+k,n]))}).map(([name,value])=>[key,name,fmt(value)])));
    metricChart();
  }
  function metricChart(){
    if(!data)return;
    const target=$('metric-chart'),name=$('metric-name').value,filter=$('metric-run').value,colors=['#2563eb','#d97706'];target.replaceChildren();$('metric-legend').replaceChildren();
    const entries=Object.entries(data.live_pear?.runs||{}),series=entries.map(([run,v],i)=>({run,color:colors[i%colors.length],points:v.history.map(row=>({step:row.step,value:row[name]??row.metrics[name]})).filter(p=>typeof p.value==='number'&&Number.isFinite(p.value))})).filter(s=>(filter==='all'||s.run===filter)&&s.points.length);
    if(!series.length){target.append(node('p','No measurements for this metric yet.','chart-empty'));return;}
    const points=series.flatMap(s=>s.points),minX=Math.min(...points.map(p=>p.step)),maxX=Math.max(minX+1,...points.map(p=>p.step)),lo=Math.min(...points.map(p=>p.value)),hi=Math.max(...points.map(p=>p.value)),pad=(hi-lo||Math.max(Math.abs(hi),1))*.08,minY=lo-pad,maxY=hi+pad;
    const width=Math.max(360,Math.min(1100,target.clientWidth)),right=width-20;
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox',`0 0 ${width} 240`);svg.setAttribute('role','img');svg.setAttribute('aria-label',name+' by optimizer step');
    function el(tag,attrs,text){const e=document.createElementNS(svg.namespaceURI,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;svg.append(e);return e;}
    const x=p=>65+(p.step-minX)/(maxX-minX)*(right-65),y=p=>205-(p.value-minY)/(maxY-minY)*185;
    for(let i=0;i<4;i++){const value=minY+(maxY-minY)*i/3,pos=y({value});el('line',{x1:65,x2:right,y1:pos,y2:pos,stroke:'#dbe3ee'});el('text',{x:58,y:pos+4,'text-anchor':'end','font-size':12,fill:'#526174'},Number(value.toPrecision(4)).toString());}
    el('text',{x:65,y:228,'font-size':12,fill:'#526174'},'Step '+minX);el('text',{x:right,y:228,'font-size':12,'text-anchor':'end',fill:'#526174'},'Step '+(maxX===minX+1?minX:maxX));
    series.forEach(s=>{const label=node('span',s.run,'metric-run-label metric-color-'+colors.indexOf(s.color));$('metric-legend').append(label);el('polyline',{points:s.points.map(p=>x(p)+','+y(p)).join(' '),fill:'none',stroke:s.color,'stroke-width':2});s.points.forEach(p=>{const circle=el('circle',{cx:x(p),cy:y(p),r:4,fill:s.color,tabindex:0,'aria-label':s.run+', step '+p.step+', '+name+': '+p.value});const show=()=>{$('metric-value').textContent=s.run+' · step '+p.step+' · '+name+' = '+p.value;};circle.addEventListener('mouseenter',show);circle.addEventListener('focus',show);});});target.append(svg);
  }
  function estimates(){
    if(!data)return;
    try{
      const perArm=Number($('gpu-budget').value)/2,mixed=$('profile').value==='mixed';
      const layout=mixed?{16:[8,4,4],32:[8,16,8],64:[16,40,8],128:[16,96,16]}:{16:[8,8,0],32:[8,24,0],64:[16,48,0],128:[16,112,0]};
      const [trainer,rollout,sim]=layout[perArm];
      $('role-budget').textContent=`Estimate per run: ${trainer} trainer + ${rollout} rollout H100s${sim?' + '+sim+' RTX simulator GPUs':''} = ${perArm} GPUs. Both modeled scenarios: ${perArm*2} GPUs. ${data.pear.allocation_note} Calculator changes do not allocate GPUs.`;
      const shared={...data.pear.default_inputs,updates:Number($('updates').value),retention:Number($('retention').value),tokens_per_node_second:Number($('throughput').value),availability:Number($('availability').value),reasoning_tokens:Number($('reasoning-tokens').value),trainer_gpus:trainer,rollout_gpus:rollout,simulator_gpus:sim,parallel_environments:rollout/4*16};
      const result={};$('estimate-results').replaceChildren();
      for(const [key,title,input]of [['moe','Existing MoE warmstart + PEAR++','moe-tokens'],['dense','Base Qwen3.8-27B + PEAR++','dense-tokens']]){
        const p={...shared,...data.pear.arm_assumptions[key],control_tokens:Number($(input).value)};
        const r=pearEstimate(p);result[key]=r;
        const card=node('article',undefined,'estimate');card.append(node('h3',title),node('strong',duration(r.wall_hours)),node('p',fmt(r.allocated_gpu_hours)+' allocated GPU-hours'+(mixed?' (H100 model workers + RTX simulation)':' (H100)'), 'cost'));
        const details=node('dl');for(const [label,value]of [['Retained episodes',fmt(r.retained_episodes)],['Attempted episodes',fmt(r.attempted_episodes)],['Generated tokens',fmt(r.generated_tokens/1e9)+' billion'],['Estimated bottleneck',r.bottleneck],['48-hour budget',fmt(Math.max(0,48-p.setup_hours)/(r.wall_hours-p.setup_hours)*p.updates)+' updates at these assumptions']])details.append(node('dt',label),node('dd',value));card.append(details);$('estimate-results').append(card);
      }
      $('parallel-total').textContent=`Together, running in parallel: ${duration(Math.max(result.moe.wall_hours,result.dense.wall_hours))} until both reach this milestone; ${fmt(result.moe.allocated_gpu_hours+result.dense.allocated_gpu_hours)} total GPU-hours. These scenarios exclude scheduler queue and the engineering/admission work still outstanding.`;
    }catch(e){$('parallel-total').textContent=e.message;}
  }
  function inventory(){const query=$('search').value.trim().toLowerCase(),source=$('source').value,role=$('role').value;filtered=data.inventory.entries.filter(r=>(!source||r.source===source)&&(!role||r.role===role)&&(!query||[r.id,r.suite,r.description,r.embodiment,r.semantic_task].join(' ').toLowerCase().includes(query)));page=Math.max(0,Math.min(page,Math.ceil(filtered.length/PAGE)-1));const rows=filtered.slice(page*PAGE,(page+1)*PAGE);table('environment-table',['Environment / task','Suite / source','Membership','Readiness'],rows.map(r=>{const tag=node('span',r.role,'tag '+(r.role.includes('evaluation')?'eval':r.role==='PEAR training candidate'?'train':'pending'));return [textCell(r.id,r.description),textCell(r.suite,r.source),tag,textCell(r.status,r.runtime)];}));$('inventory-count').textContent=fmt(filtered.length)+' matching catalog entries. No catalog row is automatically an RL admission.';$('page').textContent=filtered.length?`${page+1} / ${Math.ceil(filtered.length/PAGE)}`:'0 / 0';$('previous').disabled=page===0;$('next').disabled=(page+1)*PAGE>=filtered.length;}
  function render(value){
    data=value;if(!data.inventory)throw Error('No reasoning workspace snapshot has been imported yet.');
    $('freshness').textContent='Catalog updated '+new Date(data.updated_utc).toLocaleString();
    embodiedReasoning();simulatorFamilies();estimates();liveExperiments();
    for(const [id,key]of [['source','source'],['role','role']]){const select=$(id),old=select.value;while(select.options.length>1)select.remove(1);[...new Set(data.inventory.entries.map(r=>r[key]))].sort().forEach(v=>select.append(new Option(v,v)));select.value=old;}
    inventory();table('suite-table',['Simulator suite','Catalog entries','PEAR candidates','Evaluation entries','Interface status'],data.inventory.suites.map(r=>[r.suite,r.catalog_entries,r.pear_training_candidates,r.evaluation_entries,r.interface_status]));
    $('inventory-limits').replaceChildren(...data.inventory.limitations.map(text=>node('p',text)));
    table('timing-table',['Existing evaluation','Episodes','Median tokens / episode','Median episode wall time','Seconds / action'],Object.entries(data.timing_proxies).map(([key,r])=>[key,`${r.episodes} (${r.successes} successes)`,fmt(r.median_episode_generated_tokens),fmt(r.median_episode_seconds/60)+' minutes',fmt(r.seconds_per_action)]));
    const ul=node('ul');data.pear.estimate_limitations.forEach(s=>ul.append(node('li',s)));$('estimate-assumptions').replaceChildren(ul);$('error').hidden=true;
  }
  async function load(){try{const r=await fetch('/api/reasoning',{cache:'no-store'});if(!r.ok)throw Error('Snapshot request failed: '+r.status);render(await r.json());}catch(e){$('error').hidden=false;$('error').textContent=e.message;}}
  for(const id of ['updates','gpu-budget','profile','retention','throughput','availability','moe-tokens','dense-tokens','reasoning-tokens'])$(id).addEventListener('input',estimates);
  for(const id of ['search','source','role'])$(id).addEventListener('input',()=>{page=0;inventory();});
  for(const id of ['sim-search','sim-split'])$(id).addEventListener('input',simulatorFamilies);
  for(const id of ['metric-run','metric-name'])$(id).addEventListener('input',metricChart);
  window.addEventListener('resize',metricChart);
  $('previous').onclick=()=>{page--;inventory();};$('next').onclick=()=>{page++;inventory();};
  $('download').onclick=()=>{const fields=['id','source','suite','role','status','runtime','semantic_task','description','source_path'];const csv=[fields,...filtered.map(r=>fields.map(k=>r[k]??''))].map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=node('a');a.href=url;a.download='rpunamiya-environments.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  load();setInterval(load,15000);
}
