"use strict";
const DATA = JSON.parse(document.getElementById("experiment-data").textContent);
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const state = {cohort:"ood", method:"recorded_schedule", family:"ood", episode:"libero_goal_ood:seed61:task5:state1"};
const descriptions = {
  ood:"Both published OOD suites: ten Goal tasks and ten Spatial tasks, two new resets each. All five arms run on every case.",
  libero_goal_ood:"Goal OOD: ten tasks × two resets. The object–destination compositions differ from the original benchmark tasks.",
  libero_spatial_ood:"Spatial OOD: ten tasks × two resets. Includes object substitutions and spatially specified object selection.",
  ood_teacher_tasks:"Eight compositions had successful correction teachers: sixteen new-reset cases. This is reuse on familiar compositions.",
  ood_other_tasks:"Twelve compositions had no correction teacher. Some supplied native anchors, so this is not a fully unseen-training-task split.",
  id_panel:"Four of ten standard LIBERO10 tasks × two prescribed states. Every arm succeeds on all eight cases; this is a small retention panel."
};
const labels = {libero_goal_ood:"Goal OOD",libero_spatial_ood:"Spatial OOD",libero_10:"Standard ID"};
const rowFor = (ep, arm) => DATA.rows.find(r => r.episode_id === ep && r.arm === arm);
const percentage = (n, d) => (100*n/d).toFixed(1).replace(/\.0$/, "");
const methodButtons = DATA.arms.map((arm,i) => `<button type="button" data-method="${arm}" style="--accent:${DATA.methods[arm].color}" aria-pressed="false">0${i+1} · ${esc(DATA.methods[arm].short)}</button>`).join("");
$("method-buttons").innerHTML = methodButtons;

function paired(cohort, arm) {
  const group = DATA.groups[cohort];
  if (arm === "native") return {both_success:group.methods.native.successes,method_only_success:0,reference_only_success:0,both_failure:group.methods.native.failures};
  return group.paired.find(p => p.reference === "native" && p.method === arm);
}

function renderResults() {
  const group = DATA.groups[state.cohort];
  $("cohort-note").textContent = descriptions[state.cohort];
  $("result-cards").innerHTML = DATA.arms.map((arm,i) => {
    const method = DATA.methods[arm], result = group.methods[arm], p = paired(state.cohort,arm);
    return `<button type="button" class="result-card" data-method="${arm}" aria-pressed="${arm===state.method}" style="--accent:${method.color}"><span class="method-index">0${i+1} / ${arm==="native"?"REFERENCE":"INTERVENTION"}</span><h3>${esc(method.short)}</h3><div class="card-percent">${percentage(result.successes,result.cases)}<span style="font-size:18px">%</span></div><div class="card-fraction">${result.successes} / ${result.cases} successful cases</div><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${100*result.successes/result.cases}%"></div></div><div class="card-pair">${arm==="native"?'<span class="muted">Paired reference</span>':`<span class="gain">+${p.method_only_success} rescued</span><span class="harm">−${p.reference_only_success} lost</span>`}</div></button>`;
  }).join("");
  const result = group.methods[state.method], comparison = paired(state.cohort,state.method);
  $("paired-title").textContent = DATA.methods[state.method].short + " vs native";
  $("paired-summary").textContent = state.method === "native" ? "Native defines the reference outcome for each paired reset." : `${comparison.method_only_success} of ${group.methods.native.failures} native failures recovered; ${comparison.reference_only_success} of ${group.methods.native.successes} native successes lost.`;
  const segments = [
    ["both_success","Both succeed","#a4d0bf","#173b31"],
    ["method_only_success","Rescued","#126e56","white"],
    ["reference_only_success","Lost native success","#b84355","white"],
    ["both_failure","Both fail","#d8e0eb","#31465e"]
  ];
  $("paired-bar").innerHTML = segments.filter(([key]) => comparison[key] > 0).map(([key,label,color,foreground]) => `<span title="${label}: ${comparison[key]}" style="width:${100*comparison[key]/result.cases}%;background:${color};color:${foreground}">${comparison[key]}</span>`).join("");
  $("paired-bar").setAttribute("aria-label",segments.map(([key,label])=>`${label}: ${comparison[key]}`).join("; "));
  $("paired-legend").innerHTML = segments.map(([key,label,color])=>`<span><i style="background:${color}"></i>${label} <strong>${comparison[key]}</strong></span>`).join("");
}

function renderMethod() {
  const method = DATA.methods[state.method];
  document.querySelectorAll("#method-buttons button").forEach(button => button.setAttribute("aria-pressed",String(button.dataset.method===state.method)));
  $("method-kind").textContent = method.kind;
  $("method-kind").style.color = method.color;
  $("method-title").textContent = method.name;
  $("method-description").textContent = method.description;
  $("method-facts").innerHTML = [["Decision trigger",method.trigger],["What changes",method.changes],["Learned parameters",method.parameters],["Astra’s role",method.astra]].map(([k,v])=>`<dl><dt>${esc(k)}</dt><dd>${esc(v)}</dd></dl>`).join("");
  $("flow").innerHTML = DATA.flows[state.method];
  $("flow-download").href = `flows/${state.method}.svg`;
  $("method-training").textContent = method.training;
  $("method-reading").textContent = method.reading;
}

function setMethod(arm) {
  if (!DATA.arms.includes(arm)) return;
  state.method = arm;
  renderResults(); renderMethod();
}
document.addEventListener("click",event=>{
  const button = event.target.closest("[data-method]");
  if (button) setMethod(button.dataset.method);
});
$("cohort").addEventListener("change",event=>{state.cohort=event.target.value;renderResults();});

function renderTasks() {
  const tasks = DATA.rows.filter(r=>r.arm==="native" && r.initial_state_id===1 && (state.family==="ood" ? r.suite!=="libero_10" : r.suite===state.family));
  $("task-table").querySelector("tbody").innerHTML = tasks.map(task => {
    const tiles = DATA.arms.map(arm => `<td><div class="task-pair">${[1,2].map(reset=>{
      const row = DATA.rows.find(r=>r.arm===arm && r.suite===task.suite && r.task_id===task.task_id && r.initial_state_id===reset);
      const label = `${DATA.methods[arm].short}, reset ${reset}: ${row.credited_success?"success":"capped failure"}, ${row.actions_executed} actions. ${row.instruction}`;
      return `<button type="button" class="task-tile ${row.credited_success?"success-tile":"failure-tile"} ${row.episode_id===state.episode?"selected-case":""}" data-episode="${esc(row.episode_id)}" aria-label="${esc(label)}" title="${esc(label)}">${row.credited_success?"✓":"×"}</button>`;
    }).join("")}</div></td>`).join("");
    return `<tr><td><span class="suite-label">${labels[task.suite]} · Task ${task.task_id}</span><span class="task-name">${task.correction_teacher_task?'<i class="teacher-dot" title="Has a correction teacher"></i>':""}${esc(task.instruction)}</span></td>${tiles}</tr>`;
  }).join("");
}
$("task-filter").addEventListener("change",event=>{state.family=event.target.value;renderTasks();});
$("task-table").addEventListener("click",event=>{
  const button = event.target.closest("[data-episode]");
  if (!button) return;
  state.episode=button.dataset.episode; renderTasks();renderCase();
});

function segments(row) {
  if (row.arm==="native" || row.arm==="flow_head") return [{start:0,end:row.actions_executed,operator:row.arm==="native"?"native":"head",choice:{},probability:null}];
  return row.decisions.filter(d=>d.step<row.actions_executed).map((decision,i,decisions)=>({
    start:decision.step,end:Math.min(decisions[i+1]?.step??row.actions_executed,row.actions_executed),
    operator:row.arm==="gated_flow_head" ? (decision.gate_active?"head":"native") : decision.choice.operator,
    choice:row.arm==="gated_flow_head"?{}:decision.choice,probability:decision.gate_probability
  }));
}

function renderCase() {
  const rows = DATA.arms.map(arm=>rowFor(state.episode,arm)), reference=rows[0];
  $("case-title").textContent = reference.instruction;
  $("case-meta").textContent = `${labels[reference.suite]} · Task ${reference.task_id} · Reset ${reference.initial_state_id} · ${reference.correction_teacher_task?"Correction teacher available":"No correction teacher"} · ${reference.action_budget}-action cap`;
  $("case-metrics").innerHTML = rows.map(row=>`<div class="case-metric"><span>${esc(DATA.methods[row.arm].short)}</span><strong class="${row.credited_success?"gain":"harm"}">${row.credited_success?"✓ Success":"× Capped failure"}</strong><small>${row.actions_executed} executed actions</small></div>`).join("");
  const colors={native:"#e4e9ef",tei:"#d59735",tli:"#4274d9",head:"#845ac0"};
  $("case-timeline").innerHTML = rows.map(row=>{
    const tracks=segments(row).map(segment=>{
      let label=`Actions ${segment.start}–${segment.end}: ${segment.operator.toUpperCase()}`;
      if (segment.choice.alpha!==undefined) label+=`, α=${segment.choice.alpha.toFixed(3)}, sources ${segment.choice.source_a_id}/${segment.choice.source_b_id}`;
      if (segment.probability!==null) label+=`, gate score ${segment.probability.toFixed(3)}`;
      return `<button type="button" style="width:${100*(segment.end-segment.start)/row.action_budget}%;background:${colors[segment.operator]}" aria-label="${esc(label)}" title="${esc(label)}"></button>`;
    }).join("");
    return `<div class="timeline-row"><span>${esc(DATA.methods[row.arm].short)}</span><div class="timeline">${tracks}</div><span>${row.actions_executed} / ${row.action_budget}</span></div>`;
  }).join("");
  const hasVideo=DATA.gallery.videos.some(v=>v.episode_id===state.episode);
  $("jump-video").disabled=!hasVideo;
  $("jump-video").textContent=hasVideo?"View exported clips ↓":"No clip exported for this case";
}
const videoEpisodes=[...new Set(DATA.gallery.videos.map(v=>v.episode_id))];
$("video-case").innerHTML=videoEpisodes.map(ep=>{
  const row=rowFor(ep,"native");
  return `<option value="${esc(ep)}">${esc(labels[row.suite])} ${row.task_id}, reset ${row.initial_state_id} · ${esc(row.instruction)}</option>`;
}).join("");
$("video-case").value=state.episode;
function renderVideos() {
  const clips=DATA.gallery.videos.filter(v=>v.episode_id===$("video-case").value).sort((a,b)=>DATA.arms.indexOf(a.arm)-DATA.arms.indexOf(b.arm));
  $("video-grid").innerHTML=clips.map(clip=>`<article class="video-card"><video controls playsinline preload="metadata" src="../results/${esc(clip.path)}" aria-label="${esc(DATA.methods[clip.arm].short)}: ${esc(rowFor(clip.episode_id,clip.arm).instruction)}"></video><div class="video-caption"><h3>${esc(DATA.methods[clip.arm].short)}</h3><p><strong class="${clip.success?"gain":"harm"}">${clip.success?"Success":"Capped failure"}</strong> · ${clip.actions} actions</p><a href="../results/${esc(clip.path)}" download>Download original clip ↓</a></div></article>`).join("");
}
$("video-case").addEventListener("change",renderVideos);
$("jump-video").addEventListener("click",()=>{
  if (!videoEpisodes.includes(state.episode)) return;
  $("video-case").value=state.episode;renderVideos();$("videos").scrollIntoView({behavior:window.matchMedia("(prefers-reduced-motion: reduce)").matches?"auto":"smooth"});
});
$("gpu-hours").textContent=DATA.cost.total_l40s_gpu_hours.toFixed(2);
if (DATA.extensions) {
  const count=DATA.extensions.tasks.length;
  $("extensions").innerHTML=`<aside class="extension-panel"><span class="pill">SEPARATE BENCHMARK EXTENSION</span><h3>${count} additional OOD tasks</h3><p>New task definitions are separate from the measured twenty-task paper benchmark. No new-task success rate is included in this dashboard. <a href="../../ood_extensions/v1/index.html">Preview the tasks and validation status →</a></p></aside>`;
}
renderResults();renderMethod();renderTasks();renderCase();renderVideos();
window.dashboardState=state;
