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
  function stats(){const c=data.inventory.counts,u=data.compute_usage;const rows=[[u.allocated_gpus??'—','GPUs in current allocation'],[32,'additional H100s estimated · not launched'],[c.efe_pear_training_candidates,'EFE training candidates · not admitted'],[c.native_yam_graph_files,'native YAM graph files · '+c.native_yam_unique_env_names+' declared names']];$('summary').replaceChildren(...rows.map(([value,label])=>{const n=node('article',undefined,'stat');n.append(node('strong',fmt(value)),node('span',label));return n;}));}
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
    $('freshness').textContent='Inventory built '+new Date(data.updated_utc).toLocaleString()+'. GPU allocation has its own timestamp below.';
    stats();estimates();
    for(const [id,key]of [['source','source'],['role','role']]){const select=$(id),old=select.value;while(select.options.length>1)select.remove(1);[...new Set(data.inventory.entries.map(r=>r[key]))].sort().forEach(v=>select.append(new Option(v,v)));select.value=old;}
    inventory();table('suite-table',['Simulator suite','Catalog entries','PEAR candidates','Evaluation entries','Interface status'],data.inventory.suites.map(r=>[r.suite,r.catalog_entries,r.pear_training_candidates,r.evaluation_entries,r.interface_status]));
    $('inventory-limits').replaceChildren(...data.inventory.limitations.map(text=>node('p',text)));
    const u=data.compute_usage;$('compute-time').textContent=u.observed_utc?new Date(u.observed_utc).toLocaleString():'No allocation snapshot';
    table('compute-table',['Workflow','Pool','Allocated GPUs','State'],(u.workflows||[]).map(r=>{const link=node('a',r.name);link.href='https://us-west-2-aws.osmo.nvidia.com/workflows/'+encodeURIComponent(r.name);link.target='_blank';link.rel='noreferrer';return [link,r.pool,r.allocated_gpus??0,r.status];}));
    table('timing-table',['Existing evaluation','Episodes','Median tokens / episode','Median episode wall time','Seconds / action'],Object.entries(data.timing_proxies).map(([key,r])=>[key,`${r.episodes} (${r.successes} successes)`,fmt(r.median_episode_generated_tokens),fmt(r.median_episode_seconds/60)+' minutes',fmt(r.seconds_per_action)]));
    const ul=node('ul');data.pear.estimate_limitations.forEach(s=>ul.append(node('li',s)));$('estimate-assumptions').replaceChildren(ul);$('error').hidden=true;
  }
  async function load(){try{const r=await fetch('/api/reasoning',{cache:'no-store'});if(!r.ok)throw Error('Snapshot request failed: '+r.status);render(await r.json());}catch(e){$('error').hidden=false;$('error').textContent=e.message;}}
  for(const id of ['updates','gpu-budget','profile','retention','throughput','availability','moe-tokens','dense-tokens','reasoning-tokens'])$(id).addEventListener('input',estimates);
  for(const id of ['search','source','role'])$(id).addEventListener('input',()=>{page=0;inventory();});
  $('previous').onclick=()=>{page--;inventory();};$('next').onclick=()=>{page++;inventory();};
  $('download').onclick=()=>{const fields=['id','source','suite','role','status','runtime','semantic_task','description','source_path'];const csv=[fields,...filtered.map(r=>fields.map(k=>r[k]??''))].map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=node('a');a.href=url;a.download='rpunamiya-environments.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  load();setInterval(load,60000);
}
