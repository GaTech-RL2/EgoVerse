/* Offline viewer. Recorded text is inserted with textContent or escaped HTML. */
"use strict";
const D = window.ASTRA_DASHBOARD;
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pretty = (value) => JSON.stringify(value, null, 2);
const fmt = (value) => Number(value || 0).toLocaleString("en-US");
const compact = (value) => value >= 1e6 ? (value / 1e6).toFixed(2) + "M" : value >= 1e3 ? (value / 1e3).toFixed(1) + "K" : fmt(value);
const duration = (seconds) => seconds >= 60 ? (seconds / 60).toFixed(1) + " min" : Math.round(seconds) + " s";
let arms = ["F", "B0", "B"];
const colors = {F:"#007b73", B0:"#ac611c", B:"#7955b2", F_adaptive:"#2365d1"};
const names = {F:"Hardware interface", B0:"Source discovery", B:"Discovery + observer", F_adaptive:"Evolving hardware interface"};
const pill = (arm) => `<span class="arm-pill ${esc(arm)}">${esc(arm)}</span>`;
const status = (c) => c.outcome.success ? "Success" : c.outcome.terminal_reason === "MODEL_ERROR" ? "Model error" : c.outcome.terminal_reason === "TIMEOUT_STEPS" ? "Step horizon" : "Unsuccessful";
let current = null, eventIndex = 0, timer = null, promptCase = null, interfaceCase = null;
let cohort = Object.keys(D.cohorts)[0];
const pending = new Map();

function switchTab(name) {
  if (!$(name)?.classList.contains("tab")) return;
  document.querySelectorAll(".tab").forEach((e) => e.classList.toggle("active", e.id === name));
  document.querySelectorAll("nav button").forEach((e) => e.classList.toggle("active", e.dataset.tab === name));
  if (name !== "recorder") stopPlayback();
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
}
document.querySelectorAll("nav button").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
window.addEventListener("hashchange", () => switchTab(location.hash.slice(1)));

function loadCase(id) {
  if (window.ASTRA_RECORDINGS?.[id]) return Promise.resolve(window.ASTRA_RECORDINGS[id]);
  if (pending.has(id)) return pending.get(id);
  const promise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "recordings/" + encodeURIComponent(id) + ".js";
    script.onload = () => { const c = window.ASTRA_RECORDINGS?.[id]; c ? resolve(c) : reject(new Error("Recording did not load")); };
    script.onerror = () => reject(new Error("Recording file is missing. Unzip the entire dashboard folder."));
    document.head.append(script);
  });
  pending.set(id, promise);
  return promise;
}

function metrics(container, rows) {
  $(container).innerHTML = rows.map(([label, value, foot]) => `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="foot">${esc(foot)}</div></div>`).join("");
}
function groupCases(cases) {
  const groups = new Map();
  for (const c of cases) {
    const key = c.task + ":" + c.state;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(c);
  }
  return groups;
}
function isMatched(group) {
  const bad = ["MODEL_ERROR", "TOOL_ERROR", "PROTOCOL_DEVIATION", "RESET_FAILURE", "EVALUATOR_ERROR"];
  return group.length === arms.length && arms.every((arm) => group.filter((c) => c.condition === arm).length === 1) && new Set(group.map((c) => c.outcome.init_state_hash)).size === 1 && new Set(group.map((c) => c.outcome.env_seed)).size === 1 && group.every((c) => c.replay.status === "passed" && !bad.includes(c.outcome.terminal_reason));
}
function currentCases() { return D.cases.filter((c) => c.cohort === cohort); }

function renderOverview() {
  arms = D.cohorts[cohort].conditions || ["F", "B0", "B"];
  const cases = currentCases(), info = D.cohorts[cohort], groups = groupCases(cases);
  const complete = [...groups.values()].filter(isMatched);
  $("cohort-note").textContent = info.note;
  metrics("metrics", [["Archived recordings", cases.length + " / " + info.planned, info.stage === "smoke" ? "Short commissioning checks" : "Fresh pilot · incomplete"], ["Matched starting states", complete.length, "All assigned methods + replay passed"], ["Native control steps", fmt(cases.reduce((s,c) => s + c.outcome.sim_steps,0)), "Applied actions in these recordings"], ["Trace verification", cases.filter((c) => c.provenance.chain_verified).length + " / " + cases.length, "Archive hash + event chain verified"]]);
  $("matched-count").textContent = complete.length + " complete groups";
  $("success-chart").innerHTML = complete.length ? arms.map((arm) => {
    const n = complete.length, successes = complete.filter((g) => g.find((c) => c.condition === arm).outcome.success).length;
    return `<div class="bar-row"><span class="bar-label" style="color:${colors[arm]}">${arm}</span><div class="bar-track"><div class="bar-fill" style="width:${100*successes/n}%;background:${colors[arm]}"></div></div><span class="bar-value">${successes}/${n} · ${Math.round(100*successes/n)}%</span></div>`;
  }).join("") : '<div class="empty-state">No fully matched groups in this snapshot. These runs are not a full benchmark comparison.</div>';
  const total = (field) => cases.reduce((s,c) => s + (c.outcome[field] || 0),0);
  const retries = cases.reduce((s,c) => s + c.markers.filter((m) => m.kind === "api_retry").length,0);
  $("recording-stats").innerHTML = [["Logged tool calls",fmt(total("tool_calls"))],["Observer requests",fmt(total("observer_turns"))],["Rejected tools / accepted follow-ups",total("invalid_commands") + " / " + total("recoveries")],["Recorded API retries",retries],["Reported input + output tokens",compact(total("known_workflow_tokens"))]].map(([label,value]) => `<div class="stat-row"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join("");
  const orderedGroups = [...groups.entries()].sort((a,b) => Number(isMatched(b[1])) - Number(isMatched(a[1])) || a[0].localeCompare(b[0],undefined,{numeric:true}));
  $("group-select").innerHTML = orderedGroups.map(([key,group]) => `<option value="${esc(key)}">Task ${esc(group[0].task)} · state ${esc(group[0].state)}${isMatched(group) ? " · matched" : " · incomplete"}</option>`).join("");
  const successfulGroup = orderedGroups.find(([,g]) => isMatched(g) && g.some((c) => c.outcome.success));
  if (successfulGroup) $("group-select").value = successfulGroup[0];
  renderComparison(); renderTable();
}
function renderComparison() {
  const group = groupCases(currentCases()).get($("group-select").value) || [];
  $("comparison-instruction").textContent = group[0]?.instruction || "No trials available.";
  $("compare-grid").innerHTML = arms.map((arm) => {
    const c = group.find((v) => v.condition === arm);
    if (!c) return `<div class="compare-card"><header>${pill(arm)}<span class="muted">${names[arm]}</span></header><div class="missing">This method is not archived for the selected starting state yet.</div></div>`;
    return `<div class="compare-card"><header>${pill(arm)}<span class="${c.outcome.success ? "status-success" : "status-failure"}">${status(c)}</span></header>${c.video ? `<video controls preload="none" poster="${esc(c.thumbnail)}" src="${esc(c.video.path)}" aria-label="${arm} observation recording"></video>` : '<div class="missing">No camera recording</div>'}<div class="info"><p>${fmt(c.outcome.sim_steps)} steps · ${duration(c.outcome.wall_s)} · ${compact(c.outcome.known_workflow_tokens)} tokens</p><button class="button secondary" data-recording="${esc(c.id)}">Inspect ${arm} process →</button></div></div>`;
  }).join("");
  $("compare-grid").querySelectorAll("[data-recording]").forEach((b) => b.addEventListener("click", () => openRecording(b.dataset.recording)));
}
function renderTable() {
  const query = $("trial-search").value.toLowerCase();
  const cases = currentCases().filter((c) => [c.task,c.state,c.condition,c.instruction,c.outcome.terminal_reason].join(" ").toLowerCase().includes(query));
  $("trial-table").innerHTML = cases.map((c) => `<tr><td>Task ${esc(c.task)} · state ${esc(c.state)}<small>${esc(c.instruction.length > 77 ? c.instruction.slice(0,77) + "…" : c.instruction)}</small></td><td>${pill(c.condition)}</td><td class="${c.outcome.success ? "status-success" : "status-failure"}">${status(c)}</td><td>${fmt(c.outcome.sim_steps)}</td><td>${duration(c.outcome.wall_s)}</td><td>${compact(c.outcome.known_workflow_tokens)}</td><td><button class="chip" data-recording="${esc(c.id)}">Open →</button></td></tr>`).join("");
  $("trial-table").querySelectorAll("[data-recording]").forEach((b) => b.addEventListener("click", () => openRecording(b.dataset.recording)));
}

function stopPlayback() { if (timer) clearInterval(timer); timer = null; $("play-events").textContent = "Play decisions"; }
async function openRecording(id, show = true) {
  stopPlayback();
  if (show) switchTab("recorder");
  $("recording-select").value = id;
  $("recorder-loading").hidden = false;
  $("recorder-content").hidden = true;
  try { current = await loadCase(id); } catch(e) { $("recorder-loading").textContent = e.message; return; }
  $("recorder-loading").hidden = true; $("recorder-content").hidden = false;
  $("download-recording").href = "recordings/" + id + ".json";
  $("recorder-method").className = "arm-pill " + current.condition;
  $("recorder-method").textContent = current.condition + " · " + names[current.condition];
  $("recorder-task").textContent = current.instruction;
  $("recorder-result").textContent = current.outcome.terminal_reason;
  $("recorder-stats").innerHTML = [["Task / state",current.task + " / " + current.state],["Native steps",fmt(current.outcome.sim_steps)],["Wall time",duration(current.outcome.wall_s)],["Tool calls",fmt(current.outcome.tool_calls)],["Observer calls",current.outcome.observer_turns],["Replay",current.replay.status]].map(([k,v]) => `<span>${esc(k)}<b>${esc(v)}</b></span>`).join("");
  $("event-scrubber").max = current.events.length - 1;
  $("timeline-markers").innerHTML = current.markers.map((m) => `<button class="${esc(m.kind)}" style="left:${100*m.sequence/Math.max(1,current.events.length-1)}%" title="${esc(m.label)} · step ${m.step} · event ${m.sequence}" aria-label="${esc(m.label)} at event ${m.sequence}" data-sequence="${m.sequence}"></button>`).join("");
  $("timeline-markers").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => selectEvent(+b.dataset.sequence)));
  $("recording-video").pause();
  if (current.video) { $("recording-video").src = current.video.path; $("recording-video").hidden = false; } else { $("recording-video").removeAttribute("src"); $("recording-video").hidden = true; }
  $("evaluator-images").innerHTML = current.evaluator_images.map((im) => `<a href="${esc(im.path)}" target="_blank" rel="noopener"><img src="${esc(im.path)}" alt="${esc(im.name)}">${esc(im.name)}</a>`).join("");
  $("event-search").value = ""; $("event-filter").value = "decisions"; $("jump-status").textContent = "";
  renderEventLog();
  const firstImage = current.events.findIndex((e) => e.kind === "tool_response" && e.images.length);
  selectEvent(firstImage >= 0 ? firstImage : 0);
}

function resolvedDetail(event) {
  const result = JSON.parse(JSON.stringify(event.detail));
  if (result.request) {
    for (const key of ["instructions","tools"]) {
      const ref = result.request[key]?.dashboard_reference;
      if (ref) { const [collection,id] = ref.split("/"); result.request[key] = current[collection][id]; }
    }
  }
  return result;
}
function selectEvent(index, fromVideo = false) {
  if (!current) return;
  eventIndex = Math.max(0, Math.min(current.events.length - 1, index));
  const event = current.events[eventIndex];
  $("event-scrubber").value = eventIndex;
  $("event-position").textContent = "Event " + event.sequence + " / " + (current.events.length - 1) + " · native step " + event.step;
  $("event-wall").textContent = "Wall +" + duration(event.time);
  $("camera-step").textContent = "Current native step " + event.step;
  const latest = new Map();
  const cameras = [...new Set(current.events.flatMap((e) => e.kind === "tool_response" ? e.images.map((im) => im.camera) : []))].sort();
  for (let i=0;i<=eventIndex;i++) {
    const e = current.events[i];
    if (e.kind === "tool_response") for (const im of e.images) latest.set(im.camera, im);
  }
  $("camera-grid").innerHTML = cameras.map((camera) => {
    const im = latest.get(camera);
    return `<div class="camera-cell"><div class="camera-label">${esc(camera)}${im ? " · sampled at step " + im.step : ""}</div>${im ? `<a href="${esc(im.path)}" target="_blank" rel="noopener"><img src="${esc(im.path)}" alt="Recorded ${esc(camera)} image at step ${im.step}"></a>` : '<div class="empty">No actor observation<br>recorded yet</div>'}</div>`;
  }).join("");
  $("camera-caption").textContent = "Latest logged actor image for each camera up to this event. Cameras may have been read at different steps; each image retains its own sample step.";
  $("event-heading").textContent = "#" + event.sequence + " · " + event.kind.replaceAll("_", " ");
  $("event-summary").textContent = event.summary;
  const detail = resolvedDetail(event);
  $("event-json").textContent = pretty(detail);
  $("event-readable").replaceChildren();
  if (event.kind === "tool_request" && ["scratch_execute", "define_interface_tool"].includes(event.tool)) {
    const pre = document.createElement("pre"); pre.textContent = detail.arguments.code; $("event-readable").append(pre);
  } else if (event.kind === "model_response") {
    for (const part of detail.response.output || []) {
      if (part.type === "message") for (const content of part.content || []) if (content.type === "output_text") { const pre = document.createElement("pre"); pre.textContent = content.text; $("event-readable").append(pre); }
      if (part.type === "function_call") { const pre = document.createElement("pre"); pre.textContent = part.name + "\n" + part.arguments; $("event-readable").append(pre); }
    }
  }
  $("event-log").querySelectorAll(".log-row").forEach((row) => row.classList.toggle("selected", +row.dataset.sequence === eventIndex));
  if (!fromVideo && current.video) {
    const video = $("recording-video"); video.pause();
    const position = [...current.video.index].reverse().find((p) => p.sequence <= event.sequence);
    if (position && video.readyState) video.currentTime = position.seconds;
  }
}

function decisionIndices() {
  return current.events.filter((e) => e.kind === "tool_request" || e.kind === "trial_end" || e.kind === "error" || (e.kind === "tool_response" && (e.images.length || e.tool === "describe_visible"))).map((e) => e.sequence);
}
function moveDecision(direction) {
  if (!current) return;
  const positions = decisionIndices();
  const next = direction > 0 ? positions.find((i) => i > eventIndex) : positions.reverse().find((i) => i < eventIndex);
  selectEvent(next ?? (direction > 0 ? current.events.length - 1 : 0));
  if (eventIndex === current.events.length-1) stopPlayback();
}
function renderEventLog() {
  if (!current) return;
  const filter = $("event-filter").value, query = $("event-search").value.toLowerCase();
  const corrections = new Set(current.markers.filter((m) => ["rejection","recovery","api_retry","error"].includes(m.kind)).map((m) => m.sequence));
  const events = current.events.filter((e) => {
    if (filter === "decisions" && !["tool_request","trial_end","model_response","interface_revised","system_identification_probe"].includes(e.kind)) return false;
    if (filter === "tools" && !["tool_request","tool_response"].includes(e.kind)) return false;
    if (filter === "corrections" && !corrections.has(e.sequence)) return false;
    if (filter === "observations" && !(e.kind === "tool_response" && e.images.length)) return false;
    return !query || (e.summary + " " + JSON.stringify(e.detail)).toLowerCase().includes(query);
  });
  $("event-log").innerHTML = events.length ? events.map((e) => `<button class="log-row ${eventIndex === e.sequence ? "selected" : ""}" data-sequence="${e.sequence}"><span class="time">+${Math.round(e.time)}s</span><span class="time">step ${e.step}</span><span class="log-type">${esc(e.tool || e.kind.replaceAll("_"," "))}</span><span class="log-summary">${esc(e.summary)}</span></button>`).join("") : '<div class="empty-state">No matching events in this recording. No unrecorded attempts or retries are inferred.</div>';
  $("event-log").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => selectEvent(+b.dataset.sequence)));
}

async function renderPrompts(id) {
  promptCase = await loadCase(id);
  $("prompt-select").value = id;
  $("prompt-method").textContent = promptCase.condition + " · " + promptCase.trial_id;
  $("user-prompt").textContent = promptCase.prompts.task;
  $("actor-prompt").textContent = promptCase.prompts.actor;
  $("prompt-hash").textContent = "SHA-256 (canonical JSON string) " + promptCase.prompts.actor_sha256;
  $("tool-schemas").textContent = pretty(promptCase.prompts.tools);
  let observerCase = promptCase;
  if (!observerCase.prompts.observer) {
    const reference = D.cases.find((c) => c.cohort === promptCase.cohort && c.condition === "B" && c.outcome.observer_turns > 0) || D.cases.find((c) => c.condition === "B" && c.outcome.observer_turns > 0);
    observerCase = reference ? await loadCase(reference.id) : null;
  }
  $("observer-prompt").textContent = observerCase?.prompts.observer || "No observer request is available in this snapshot.";
  $("observer-prompt-source").textContent = observerCase ? "Recorded in " + observerCase.trial_id : "Not available";
  $("observer-format").textContent = pretty(observerCase?.prompts.observer_format || null);
}

async function renderInterface(id) {
  interfaceCase = await loadCase(id);
  $("interface-select").value = id;
  $("example-instruction").textContent = "Recorded example from " + interfaceCase.trial_id + ". Episode IDs, numbers, timestamps and results are from the archive.";
  const labels = [...(interfaceCase.device ? ["Device description"] : []), ...Object.keys(interfaceCase.examples)];
  $("example-tabs").innerHTML = labels.map((label) => `<button class="chip" data-example="${esc(label)}">${esc(label)}</button>`).join("");
  $("example-tabs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => renderExample(b.dataset.example)));
  renderExample(labels.includes("act") ? "act" : labels.includes("step") ? "step" : labels[0]);
}
function renderExample(label) {
  $("example-tabs").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.example === label));
  $("example-content").replaceChildren();
  if (label === "Device description") {
    const pre = document.createElement("pre"); pre.textContent = pretty(interfaceCase.device); $("example-content").append(pre); return;
  }
  const example = interfaceCase.examples[label];
  if (!example) return;
  const grid = document.createElement("div"); grid.className = "example-columns";
  for (const [title,value] of [["Arguments sent",example.arguments],["Returned tool result",example.response ?? "No tool return was recorded for this call."]]) {
    const col = document.createElement("div"), heading = document.createElement("h3"), pre = document.createElement("pre"); heading.textContent = title; pre.textContent = pretty(value); col.append(heading,pre); grid.append(col);
  }
  const note = document.createElement("p"); note.className = "example-note"; note.textContent = "Original event #" + example.sequence + ". PNG fields link to exact local image files. This example is a recorded call; check its acceptance/result before reusing it.";
  const link = document.createElement("button"); link.className = "chip"; link.textContent = "Inspect this event →"; link.addEventListener("click", async () => { await openRecording(interfaceCase.id); selectEvent(example.sequence); });
  $("example-content").append(grid,note,link);
}

async function copyText(text, button) {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else { const textarea = document.createElement("textarea"); textarea.value = text; document.body.append(textarea); textarea.select(); document.execCommand("copy"); textarea.remove(); }
    const old = button.textContent; button.textContent = "Copied"; setTimeout(() => { button.textContent = old; },1200);
  } catch { button.textContent = "Select text below to copy"; }
}

$("cohort-select").innerHTML = Object.entries(D.cohorts).filter(([key]) => D.cases.some((c) => c.cohort === key)).map(([key,info]) => `<option value="${esc(key)}">${esc(info.title)}</option>`).join("");
$("cohort-select").addEventListener("change", () => { cohort = $("cohort-select").value; renderOverview(); });
$("group-select").addEventListener("change",renderComparison);
$("trial-search").addEventListener("input",renderTable);
for (const id of ["recording-select","prompt-select","interface-select"]) $(id).innerHTML = D.cases.map((c) => `<option value="${esc(c.id)}">${esc(D.cohorts[c.cohort].benchmark)} · ${esc(c.trial_id)} · ${status(c)}</option>`).join("");
$("recording-select").addEventListener("change", () => openRecording($("recording-select").value));
$("prompt-select").addEventListener("change", () => renderPrompts($("prompt-select").value));
$("interface-select").addEventListener("change", () => renderInterface($("interface-select").value));
$("copy-prompt").addEventListener("click", () => copyText(promptCase.prompts.actor,$("copy-prompt")));
$("copy-event").addEventListener("click", () => copyText($("event-json").textContent,$("copy-event")));
$("previous-event").addEventListener("click",() => { stopPlayback(); moveDecision(-1); });
$("next-event").addEventListener("click",() => { stopPlayback(); moveDecision(1); });
$("event-scrubber").addEventListener("input",() => { stopPlayback(); selectEvent(+$("event-scrubber").value); });
$("play-events").addEventListener("click",() => { if (timer) stopPlayback(); else { $("recording-video").pause(); $("play-events").textContent = "Pause"; timer = setInterval(() => moveDecision(1),+$("play-speed").value); } });
$("play-speed").addEventListener("change",() => { if (timer) { clearInterval(timer); timer = setInterval(() => moveDecision(1),+$("play-speed").value); } });
$("event-filter").addEventListener("change",renderEventLog);
let searchTimer;
$("event-search").addEventListener("input",() => { clearTimeout(searchTimer); searchTimer = setTimeout(renderEventLog,150); });
document.querySelectorAll("[data-jump]").forEach((button) => button.addEventListener("click",() => {
  if (!current) return;
  stopPlayback(); const kind = button.dataset.jump;
  let candidates;
  if (kind === "observation") candidates = current.events.filter((e) => e.kind === "tool_response" && e.images.length).map((e) => ({sequence:e.sequence,label:"Recorded actor observation"}));
  else candidates = current.markers.filter((m) => kind === "correction" ? ["rejection","recovery","api_retry","error"].includes(m.kind) : m.kind === kind);
  const next = candidates.find((m) => m.sequence > eventIndex) || candidates[0];
  if (next) { const wrapped = next.sequence <= eventIndex; selectEvent(next.sequence); $("jump-status").textContent = (wrapped ? "Wrapped to first marker: " : "") + next.label + " · event #" + next.sequence; }
  else $("jump-status").textContent = "No recorded " + kind + " marker in this trial.";
}));
$("recording-video").addEventListener("play",stopPlayback);
$("recording-video").addEventListener("timeupdate",() => {
  if (!current?.video || $("recording-video").paused) return;
  const position = [...current.video.index].reverse().find((p) => p.seconds <= $("recording-video").currentTime);
  if (position && position.sequence !== eventIndex) selectEvent(position.sequence,true);
});

const stamp = new Date(D.catalog_utc).toLocaleString("en-US",{month:"short",day:"numeric",hour:"numeric",minute:"2-digit",timeZoneName:"short"});
$("snapshot-date").textContent = stamp;
$("footer-stamp").textContent = "Archive snapshot: " + D.catalog_utc.replace("T"," ");
metrics("evidence-metrics", [["Recordings",D.cases.length,"Archived terminated trials"],["Event chains verified",D.cases.filter((c) => c.provenance.chain_verified).length,"Every sequence and event digest"],["Replay receipts passed",D.cases.filter((c) => c.replay.status === "passed").length,"Independent native action replay"],["Recorded events",fmt(D.cases.reduce((s,c) => s+c.provenance.event_count,0)),"Full event order remains inspectable"]]);
$("source-provenance").innerHTML = Object.entries(D.cohorts).map(([key,info]) => `<div class="provenance-card"><strong>${esc(info.title)}</strong><p>Workflow: <a href="https://us-west-2-aws.osmo.nvidia.com/workflows/${esc(key)}" target="_blank" rel="noopener">${esc(key)}</a></p><p>Source commit: <code>${esc(info.commit)}</code></p><p>${esc(info.note)}</p></div>`).join("");
renderOverview();
const defaultCase = D.cases.find((c) => c.condition === "F" && c.outcome.invalid_commands > 0) || D.cases[0];
const defaultPrompt = D.cases.find((c) => c.condition === "B0") || D.cases[0];
const defaultInterface = D.cases.find((c) => c.condition === "F") || D.cases[0];
if (defaultCase) { openRecording(defaultCase.id,false); renderPrompts(defaultPrompt.id); renderInterface(defaultInterface.id); }
switchTab(location.hash.slice(1) || "overview");
