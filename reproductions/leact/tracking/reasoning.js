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
    const r=data.embodied_reasoning||{},families=r.families||[],admitted=families.reduce((n,f)=>n+(f.admitted_tasks||0),0);
    $('reasoning-status').textContent=`${families.length} planned families · ${admitted} admitted tasks`;
    table('reasoning-task-table',['Training family','Tasks and output','Evaluation targets'],families.map(f=>{
      const name=textCell(f.name,f.status==='planned_adapter_and_generator_not_implemented'?'Generator not implemented':f.status);
      const tasks=node('div'),example=node('p',f.example,'reasoning-example');tasks.append(node('div',f.output),example);
      const details=node('details'),summary=node('summary','Task list and scene sources'),list=node('ul');f.tasks.forEach(t=>list.append(node('li',t)));details.append(summary,list,node('p',f.training_environment));tasks.append(details);
      return [name,tasks,(f.transfer_targets||[]).join(', ')];
    }));
    $('reasoning-benchmarks').replaceChildren(...(r.evaluation_benchmarks||[]).map(name=>node('span',name,'tag eval')));
    $('reasoning-bbox-status').textContent=(r.box_output_planned?'Bounding-box outputs are specified in the proposal.':'Pointing uses normalized image points. Separate bounding-box prediction is not yet specified in the training proposal.')+' Named benchmark datasets remain evaluation-only; generator implementation and scene admission are pending.';
  }
  function taskGroup(title,tasks,suite,kind){
    const group=node('section',undefined,'task-group '+kind);group.append(node('h4',title+' · '+fmt(tasks.length)));
    const list=node('ul');list.tabIndex=0;list.setAttribute('aria-label',title+' tasks');
    for(const task of tasks){const label=task.id.replace(suite+'-','').replace(/-(standard-realtime|standard|privileged)$/,'').replace(/-i(\d+)$/,' · variant $1').replaceAll('_',' ').replaceAll('-',' '),item=node('li',label);item.title=task.id;item.dataset.taskId=task.id;list.append(item);}
    group.append(list);return group;
  }
  function simulatorFamilies(){
    const all=data.simulators?.families||[],query=$('sim-search').value.trim().toLowerCase(),split=$('sim-split').value;
    const families=all.filter(f=>(split==='all'||f[split]?.length)&&(!query||[f.name,f.id,...['train','eval_seen','eval_unseen','pending'].flatMap(k=>(f[k]||[]).map(t=>t.id+' '+(t.description||'')))].join(' ').toLowerCase().includes(query)));
    const counts=data.simulators?.counts||{};
    $('mixture-summary').textContent=`${fmt(counts.train||0)} train candidates · ${fmt(counts.eval_seen||0)} seen-sim / unseen-task eval tasks · ${fmt(counts.eval_unseen||0)} unseen-sim / unseen-task eval tasks`;
    $('simulator-count').textContent=fmt(families.length)+' families';
    const cards=families.map(f=>{
      const card=node('article',undefined,'family-card');
      if(f.image){const figure=node('figure'),img=node('img');img.src=f.image.url;img.alt=f.name+' simulator preview';img.loading='lazy';img.decoding='async';figure.append(img);const caption=node('figcaption'),source=node('a',f.image.caption);source.href=f.image.source_url;source.target='_blank';source.rel='noreferrer';caption.append(source);figure.append(caption);card.append(figure);}
      const body=node('div',undefined,'family-body'),heading=node('div',undefined,'family-heading');heading.append(node('h3',f.name),node('span',f.split_status,'tag '+(f.eval_unseen.length?'eval':f.pending.length?'pending':'train')));body.append(heading);
      if(!f.pending.length)body.append(taskGroup('Train',f.train,f.id,'training'));
      if(f.eval_seen.length)body.append(taskGroup('Eval · seen sim, unseen task',f.eval_seen,f.id,'evaluation'));
      if(f.eval_unseen.length)body.append(taskGroup('Eval · unseen sim, unseen task',f.eval_unseen,f.id,'evaluation'));
      if(f.pending.length)body.append(taskGroup('Catalog · train/eval split pending',f.pending,f.id,'pending'));
      if(!f.eval_seen.length&&!f.eval_unseen.length&&!f.pending.length)body.append(node('p','No eval tasks in these two categories.','meta'));
      card.append(body);return card;
    });
    $('simulator-families').replaceChildren(...cards);if(!cards.length)$('simulator-families').append(node('p','No matching simulator families.','meta'));
  }
  function estimates(){
    if(!data)return;
    try{
      const perArm=Number($('gpu-budget').value)/2,mixed=$('profile').value==='mixed';
      const layout=mixed?{16:[8,4,4],32:[8,16,8],64:[16,40,8],128:[16,96,16]}:{16:[8,8,0],32:[8,24,0],64:[16,48,0],128:[16,112,0]};
      const [trainer,rollout,sim]=layout[perArm];
      $('role-budget').textContent=`Per run: ${trainer} trainer + ${rollout} rollout + ${sim} simulator GPUs = ${perArm} H100s. Both runs: ${perArm*2} additional H100s. What-if settings do not change the 32-GPU estimate-only instruction.`;
      const shared={...data.pear.default_inputs,updates:Number($('updates').value),retention:Number($('retention').value),tokens_per_node_second:Number($('throughput').value),availability:Number($('availability').value),reasoning_tokens:Number($('reasoning-tokens').value),trainer_gpus:trainer,rollout_gpus:rollout,simulator_gpus:sim,parallel_environments:rollout/4*16};
      const result={};$('estimate-results').replaceChildren();
      for(const [key,title,input]of [['moe','Existing MoE warmstart + PEAR++','moe-tokens'],['dense','Base Qwen3.8-27B + PEAR++','dense-tokens']]){
        const p={...shared,...data.pear.arm_assumptions[key],control_tokens:Number($(input).value)};
        const r=pearEstimate(p);result[key]=r;
        const card=node('article',undefined,'estimate');card.append(node('h3',title),node('strong',duration(r.wall_hours)),node('p',fmt(r.allocated_gpu_hours)+' allocated H100 GPU-hours','cost'));
        const details=node('dl');for(const [label,value]of [['Retained episodes',fmt(r.retained_episodes)],['Attempted episodes',fmt(r.attempted_episodes)],['Generated tokens',fmt(r.generated_tokens/1e9)+' billion'],['Estimated bottleneck',r.bottleneck],['48-hour budget',fmt(Math.max(0,48-p.setup_hours)/(r.wall_hours-p.setup_hours)*p.updates)+' updates at these assumptions']])details.append(node('dt',label),node('dd',value));card.append(details);$('estimate-results').append(card);
      }
      $('parallel-total').textContent=`Together, running in parallel: ${duration(Math.max(result.moe.wall_hours,result.dense.wall_hours))} until both reach this milestone; ${fmt(result.moe.allocated_gpu_hours+result.dense.allocated_gpu_hours)} total GPU-hours. These scenarios exclude scheduler queue and the engineering/admission work still outstanding.`;
    }catch(e){$('parallel-total').textContent=e.message;}
  }
  function inventory(){const query=$('search').value.trim().toLowerCase(),source=$('source').value,role=$('role').value;filtered=data.inventory.entries.filter(r=>(!source||r.source===source)&&(!role||r.role===role)&&(!query||[r.id,r.suite,r.description,r.embodiment,r.semantic_task].join(' ').toLowerCase().includes(query)));page=Math.max(0,Math.min(page,Math.ceil(filtered.length/PAGE)-1));const rows=filtered.slice(page*PAGE,(page+1)*PAGE);table('environment-table',['Environment / task','Suite / source','Membership','Readiness'],rows.map(r=>{const tag=node('span',r.role,'tag '+(r.role.includes('evaluation')?'eval':r.role==='PEAR training candidate'?'train':'pending'));return [textCell(r.id,r.description),textCell(r.suite,r.source),tag,textCell(r.status,r.runtime)];}));$('inventory-count').textContent=fmt(filtered.length)+' matching catalog entries. No catalog row is automatically an RL admission.';$('page').textContent=filtered.length?`${page+1} / ${Math.ceil(filtered.length/PAGE)}`:'0 / 0';$('previous').disabled=page===0;$('next').disabled=(page+1)*PAGE>=filtered.length;}
  function render(value){
    data=value;if(!data.inventory)throw Error('No reasoning workspace snapshot has been imported yet.');
    $('freshness').textContent='Updated '+new Date(data.updated_utc).toLocaleString();
    embodiedReasoning();simulatorFamilies();estimates();
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
  $('previous').onclick=()=>{page--;inventory();};$('next').onclick=()=>{page++;inventory();};
  $('download').onclick=()=>{const fields=['id','source','suite','role','status','runtime','semantic_task','description','source_path'];const csv=[fields,...filtered.map(r=>fields.map(k=>r[k]??''))].map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=node('a');a.href=url;a.download='rpunamiya-environments.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  load();setInterval(load,60000);
}
