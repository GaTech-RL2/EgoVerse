const $ = id => document.getElementById(id);
const SUPERSEDED = 4001;   // matches rollout_dashboard.SUPERSEDED_CLOSE_CODE
const cameras = new Map();
let socket;
let expectedImages = [];
let retryDelay = 500;
let overlayCamera;
let paused = false;
let started = false;
let waitForStart = false;
let videoEnabled = false;
let videoRecording = false;
let selectedVideo;
let episodeEnabled = false;
let episodeRecording = false;
let episodeFrames = 0;
let episodeRequestPending = false;
let episodeLastSaved;
let episodeRequestTimer;
let modelBrowserEnabled = false;
let currentCheckpoint;
let modelDirectory = '.';
let modelLoadRevision = 0;
let modelSearchTimer;
let inferenceControls = {};
let inferenceControlSignature = '';
let inferenceControlRevision = 0;
let inferenceDrafts = {};
let pendingInferenceApply;
let inferenceApplyFeedback = '';
let ikRelaxationPending = null;
let ikSolverPending = null;
let ikSolverApplied = 'mujoco';

// Generated from the teleop shared renderer by build_reset_guides.py.
// BEGIN SHARED RESET GUIDE RENDERER
const resetGuideColors = ['#62d5d0', '#ffd06a', '#c6a0ff', '#a7dc7e'];
const resetGuideNote = 'Visual guide only · not recorded · no robot motion';
let resetTemplates = [];

function validateResetCatalog(catalog) {
  if (catalog?.version !== 1 || !Array.isArray(catalog.templates)) throw new Error('Unsupported reset template catalog');
  const ids = new Set();
  for (const template of catalog.templates) {
    if (!template || typeof template.id !== 'string' || !template.id || ids.has(template.id)
        || typeof template.label !== 'string' || !template.label
        || typeof template.camera !== 'string' || !template.camera
        || !Array.isArray(template.image_size) || template.image_size.length !== 2
        || !template.image_size.every(n => Number.isInteger(n) && n > 0)
        || !Array.isArray(template.regions) || !template.regions.length) throw new Error('Invalid reset template');
    ids.add(template.id);
    for (const region of template.regions) {
      if (!region || typeof region.label !== 'string' || !region.label
          || !Array.isArray(region.points) || region.points.length < 3
          || !region.points.every(point => Array.isArray(point) && point.length === 2
            && point.every((n, axis) => Number.isFinite(n) && n >= 0 && n <= template.image_size[axis]))) {
        throw new Error('Invalid reset template region');
      }
    }
  }
  return catalog.templates;
}

function selectedResetTemplate() {
  return resetTemplates.find(template => template.id === $('reset-guide')?.value);
}

function resetGuideVisibility() {
  const select = $('reset-guide');
  if (!select || select.disabled) return; // Safe during a static-only rolling update.
  const template = selectedResetTemplate();
  for (const tile of cameras.values()) tile.overlay.setAttribute('hidden', '');
  let status = resetGuideNote;
  if (template) {
    const tile = cameras.get(template.camera);
    if (!tile) status = `Waiting for ${template.camera} · ${resetGuideNote}`;
    else if (!tile.frameAvailable || !tile.imageReady) status = `Waiting for a live frame · ${resetGuideNote}`;
    else {
      const [width, height] = template.image_size;
      // One display-only renderer for both <img> and decoded-frame <canvas> views.
      const imageWidth = tile.image.naturalWidth ?? tile.image.width;
      const imageHeight = tile.image.naturalHeight ?? tile.image.height;
      const aspectError = Math.abs(imageWidth / imageHeight / (width / height) - 1);
      if (!Number.isFinite(aspectError) || aspectError > 0.005) {
        status = `Guide hidden: camera aspect ratio differs from template · ${resetGuideNote}`;
      } else {
        tile.overlay.removeAttribute('hidden');
        status = `${template.camera} · recheck placement if the camera moved · ${resetGuideNote}`;
      }
    }
  }
  $('reset-guide-status').textContent = status;
}

function renderResetGuide() {
  for (const tile of cameras.values()) tile.overlay.replaceChildren();
  const template = selectedResetTemplate();
  const tile = template && cameras.get(template.camera);
  if (tile) {
    tile.overlay.setAttribute('viewBox', `0 0 ${template.image_size.join(' ')}`);
    for (const [index, region] of template.regions.entries()) {
      const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      polygon.setAttribute('points', region.points.map(point => point.join(',')).join(' '));
      polygon.setAttribute('stroke', resetGuideColors[index % resetGuideColors.length]);
      polygon.setAttribute('fill', resetGuideColors[index % resetGuideColors.length]);
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      // Label centroid works for any polygon; no task-specific square assumption.
      label.setAttribute('x', region.points.reduce((sum, p) => sum + p[0], 0) / region.points.length);
      label.setAttribute('y', region.points.reduce((sum, p) => sum + p[1], 0) / region.points.length + 4);
      label.textContent = region.label;
      tile.overlay.append(polygon, label);
    }
  }
  resetGuideVisibility();
}

function setupResetGuides() {
  const select = $('reset-guide');
  if (!select) return; // An older cached page must retain its normal controls.
  try {
    resetTemplates = validateResetCatalog(JSON.parse($('reset-guide-catalog').textContent));
    for (const template of resetTemplates) {
      const option = document.createElement('option');
      option.value = template.id;
      option.textContent = template.label;
      select.append(option);
    }
    select.value = ''; // Deliberately Off on page load; selection survives WS reconnects.
    select.onchange = renderResetGuide;
    select.onkeydown = event => event.stopPropagation();
  } catch (error) {
    resetTemplates = [];
    select.disabled = true;
    $('reset-guide-status').textContent = `Reset guides unavailable: ${error.message}. Camera preview is unaffected.`;
  }
}
// END SHARED RESET GUIDE RENDERER

function updateIkSolver(settings) {
  const available = Boolean(settings?.available);
  $('ik-solver-control').hidden = !available;
  ikSolverApplied = settings?.backend ?? 'mujoco';
  if (ikSolverApplied === ikSolverPending || settings?.error) ikSolverPending = null;
  $('ik-solver').value = ikSolverPending ?? ikSolverApplied;
  $('ik-solver').disabled = !available || ikSolverPending !== null;
  $('ik-solver-detail').textContent = settings?.error ?? (ikSolverPending ? 'Applying…' : '');
}

$('ik-solver').addEventListener('change', event => {
  if (send({ik_solver: event.target.value})) {
    ikSolverPending = event.target.value;
    event.target.disabled = true;
  } else {
    event.target.value = ikSolverApplied;
    reportDisconnected();
  }
});

function updateIkRelaxation(settings) {
  const available = Boolean(settings?.available);
  $('ik-relaxation-control').hidden = !available;
  if (settings?.enabled === ikRelaxationPending) ikRelaxationPending = null;
  $('ik-relaxation').checked = ikRelaxationPending ?? Boolean(settings?.enabled);
  $('ik-relaxation').disabled = !available || ikRelaxationPending !== null;
  $('ik-relaxation-detail').textContent = available
    ? ` (up to ${(settings.position_tolerance_m * 1000).toFixed(1)} mm / ${settings.rotation_tolerance_deg}°${ikRelaxationPending !== null ? '; applying…' : ''})`
    : '';
}

function updateGripperForce(limits) {
  const entries = Object.entries(limits ?? {});
  const root = $('gripper-controls');
  $('gripper-settings').hidden = entries.length === 0;
  if (root.childElementCount !== entries.length) {
    root.replaceChildren();
    for (const [arm] of entries) {
      const label = document.createElement('label');
      label.className = 'gripper-control';
      label.dataset.arm = arm;
      const name = document.createElement('span');
      name.textContent = `${arm} arm`;
      const slider = document.createElement('input');
      slider.type = 'range'; slider.min = '1'; slider.max = '50'; slider.step = '1';
      const readout = document.createElement('output');
      slider.oninput = () => { readout.textContent = `${slider.value} N`; };
      slider.onchange = () => {
        if (!send({gripper_force: {arm, value: Number(slider.value)}})) reportDisconnected();
      };
      label.append(name, slider, readout);
      root.append(label);
    }
  }
  for (const [arm, value] of entries) {
    const slider = root.querySelector(`[data-arm="${arm}"] input`);
    if (slider && document.activeElement !== slider) {
      slider.value = value;
      slider.nextElementSibling.textContent = `${Math.round(value)} N`;
    }
  }
}

$('ik-relaxation').addEventListener('change', event => {
  const enabled = event.target.checked;
  if (send({ik_relaxation: enabled})) {
    ikRelaxationPending = enabled;
    event.target.disabled = true;
  } else {
    event.target.checked = !enabled;
    reportDisconnected();
  }
});

function send(message) {
  if (socket?.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(message));
  return true;
}

function reportDisconnected() {
  $('status').textContent = 'Dashboard is not connected yet; wait for Ready, then press c.';
  $('status').className = 'starting';
}

function stopRollout() {
  if (confirm('Stop this rollout? The robot will follow the existing rollout shutdown path.')) {
    if (!send({stop: true})) {
      reportDisconnected();
      return;
    }
    $('stop').disabled = true;
    $('stop').textContent = 'Stopping rollout…';
  }
}

function startRollout() {
  if (!send({start: true})) {
    reportDisconnected();
    return;
  }
  $('start').disabled = true;
  $('start').textContent = 'Starting…';
}

function restartRollout() {
  if (!send({restart: true})) {
    reportDisconnected();
    return false;
  }
  $('start').disabled = false;
  $('start').textContent = 'Start rollout (c)';
  return true;
}

function reconnectCameras() {
  if (!confirm('Reconnect all configured RGB cameras? Rollout control will pause and require c to resume.')) return;
  if (!send({reconnect_cameras: true})) {
    reportDisconnected();
    return;
  }
  $('status').textContent = 'Reconnecting RGB cameras; replug the camera, then wait for Ready.';
  $('status').className = 'starting';
}

function updatePauseButton() {
  $('pause').disabled = !started;
  $('pause').textContent = paused ? 'Resume rollout (Space)' : 'Pause rollout (Space)';
}

function updateRecordingIndicator() {
  const parts = [];
  if (videoRecording) parts.push('VIDEO');
  if (episodeRecording) parts.push(`EPISODE · ${episodeFrames} frames`);
  $('recording-indicator').hidden = parts.length === 0;
  $('recording-indicator').textContent = `● RECORDING ${parts.join(' + ') || 'VIDEO'}`;
}

function updateVideoControls() {
  updateRecordingIndicator();
  $('record-video').disabled = !videoEnabled || (!started && !videoRecording);
  $('record-video').textContent = videoRecording ? 'Save video (v)' : 'Record video (v)';
  $('open-videos').disabled = !videoEnabled;
}

function describeEpisode(episode) {
  const state = episode.complete ? 'complete' : `incomplete (${episode.end_reason})`;
  return `${episode.id} · ${episode.outcome} · ${episode.frames} frames · ${formatDuration(episode.duration_seconds)} · ${state}`;
}

function updateEpisodeControls() {
  updateRecordingIndicator();
  const button = $('record-episode');
  button.disabled = !episodeEnabled || episodeRecording || !started || episodeRequestPending;
  button.textContent = episodeRequestPending && !episodeRecording
    ? 'Starting episode…' : 'Record episode (d)';
  $('open-episodes').disabled = !episodeEnabled;
  $('episode-save').hidden = !episodeRecording;
  for (const save of document.querySelectorAll('#episode-save button')) {
    save.disabled = episodeRequestPending;
  }
  $('episode-title').textContent = `Recording episode · ${episodeFrames} frames`;
  const last = $('episode-status');
  last.hidden = episodeRecording || !episodeLastSaved;
  if (episodeLastSaved) last.textContent = `Last episode saved: ${describeEpisode(episodeLastSaved)}`;
}

function updateModelControl() {
  $('select-model').disabled = !modelBrowserEnabled;
}

function updateCurrentModel() {
  const text = currentCheckpoint
    ? `Selected model: ${currentCheckpoint}`
    : 'Selected model: unavailable';
  $('current-model').textContent = text;
  $('current-model').title = text;
}

async function loadModels(path = '.', query = $('model-search').value) {
  if (!modelBrowserEnabled) return;
  const revision = ++modelLoadRevision;
  const list = $('model-list');
  list.textContent = 'Loading…';
  try {
    const parameters = new URLSearchParams({path, query});
    const response = await fetch(`/api/checkpoints?${parameters}`, {cache: 'no-store'});
    if (!response.ok) throw Error(`Could not load checkpoint directory (${response.status})`);
    const listing = await response.json();
    if (revision !== modelLoadRevision) return;
    modelDirectory = listing.path;
    $('model-path').textContent = `Folder: ${listing.path}`;
    $('model-up').disabled = listing.parent === null;
    list.replaceChildren();
    for (const entry of listing.entries) {
      const row = document.createElement('button');
      row.className = `model-row ${entry.type}`;
      const kind = document.createElement('span');
      kind.className = 'model-kind';
      kind.textContent = entry.type === 'directory' ? 'Folder' : 'Checkpoint';
      const name = document.createElement('span');
      name.className = 'model-name';
      name.textContent = entry.name;
      row.append(kind, name);
      if (entry.type === 'directory') {
        row.onclick = () => {
          $('model-search').value = '';
          loadModels(entry.path, '');
        };
      } else {
        row.onclick = () => selectModel(entry);
      }
      list.append(row);
    }
    if (!listing.entries.length) {
      list.textContent = query.trim()
        ? 'No folders or .ckpt files contain that text.'
        : 'No folders or .ckpt files here.';
    }
    $('model-up').onclick = () => {
      if (listing.parent === null) return;
      $('model-search').value = '';
      loadModels(listing.parent, '');
    };
  } catch (error) {
    if (revision !== modelLoadRevision) return;
    list.textContent = error.message;
  }
}

function selectModel(entry) {
  if (!confirm(`Load ${entry.name}? Existing plan actions will be discarded, the robot will hold position, and you must press Start after loading.`)) return;
  if (!send({select_model: entry.path})) {
    reportDisconnected();
    return;
  }
  $('models').close();
  $('select-model').disabled = true;
  $('status').textContent = `Loading ${entry.name}; rollout control is paused.`;
  $('status').className = 'starting';
}

function toggleVideoRecording() {
  if (!videoEnabled || (!started && !videoRecording)) return;
  if (!send({record_video: true})) {
    reportDisconnected();
    return;
  }
  $('record-video').disabled = true;
  $('record-video').textContent = videoRecording ? 'Saving video…' : 'Starting video…';
}

function startEpisode() {
  if (!episodeEnabled || episodeRecording || !started) return;
  if (!send({record_episode: 'start'})) {
    reportDisconnected();
    return;
  }
  markEpisodeRequestPending();
}

function saveEpisode(outcome) {
  if (!episodeRecording) return;
  if (!send({save_episode: outcome})) {
    reportDisconnected();
    return;
  }
  markEpisodeRequestPending();
}

function discardEpisode() {
  if (!episodeRecording) return;
  if (!confirm('Discard this episode? Its HDF5 file is deleted.')) return;
  if (!send({discard_episode: true})) {
    reportDisconnected();
    return;
  }
  markEpisodeRequestPending();
}

function markEpisodeRequestPending() {
  // A request the rollout loop declines (no start yet, low disk) changes no
  // recording state, so the pending state also clears on its own.
  episodeRequestPending = true;
  clearTimeout(episodeRequestTimer);
  episodeRequestTimer = setTimeout(() => {
    episodeRequestPending = false;
    updateEpisodeControls();
  }, 3000);
  updateEpisodeControls();
}

function toggleEpisode() {
  if (episodeRecording) saveEpisode('unlabeled');
  else startEpisode();
}

function applyEpisodeState(message) {
  const recording = Boolean(message.episode_recording);
  if (recording !== episodeRecording) episodeRequestPending = false;
  episodeRecording = recording;
  episodeFrames = Number(message.episode_frames) || 0;
  if (message.episode_last_saved) episodeLastSaved = message.episode_last_saved;
}

function togglePause() {
  if (!send({paused: !paused})) reportDisconnected();
}

function parsedInferenceDraft(name) {
  const spec = inferenceControls[name];
  const raw = inferenceDrafts[name] ?? '';
  let value = raw, valid = false;
  if (spec?.type === 'boolean') {
    valid = raw === 'true' || raw === 'false';
    value = raw === 'true';
  } else if (spec?.type === 'enum') {
    valid = spec.choices.includes(raw);
  } else if (spec) {
    value = Number(raw);
    const steps = spec.step == null ? 0 : (value - spec.min) / spec.step;
    valid = raw.trim() !== '' && Number.isFinite(value)
      && (spec.type === 'number' || Number.isInteger(value))
      && value >= spec.min && value <= spec.max
      && Math.abs(steps - Math.round(steps)) < 1e-8;
  }
  return {valid, value, dirty: valid && value !== spec?.value};
}

function updateInferenceApplyState() {
  const button = $('apply-inference');
  const status = $('inference-apply-status');
  if (!button || !status) return;
  const parsed = Object.keys(inferenceControls).map(parsedInferenceDraft);
  const invalid = parsed.some(item => !item.valid);
  const dirty = parsed.some(item => item.dirty);
  button.disabled = Boolean(pendingInferenceApply) || invalid || !dirty
    || socket?.readyState !== WebSocket.OPEN;
  button.textContent = pendingInferenceApply ? 'Queued…' : 'Apply settings';
  if (pendingInferenceApply) {
    status.textContent = started && !paused
      ? 'Queued; the current action prefix will finish before these settings apply.'
      : 'Applying settings before the next plan.';
  } else if (invalid) {
    status.textContent = 'Enter values within the limits declared by this model profile.';
  } else if (dirty) {
    status.textContent = 'Unapplied changes. Press Enter or Apply settings.';
  } else {
    status.textContent = inferenceApplyFeedback
      || 'Active values from the selected model profile.';
  }
}

// Widget type and validation are owned by the selected model's YAML profile.
function editInferenceControl(name, input) {
  inferenceDrafts[name] = input.value;
  inferenceApplyFeedback = '';
  const parsed = parsedInferenceDraft(name);
  input.setCustomValidity(parsed.valid ? '' : 'Use a valid value in the declared range.');
  input.dataset.dirty = String(parsed.dirty);
  updateInferenceApplyState();
}

function applyInferenceOverrides() {
  if (pendingInferenceApply) return;
  const values = {};
  for (const name of Object.keys(inferenceControls)) {
    const parsed = parsedInferenceDraft(name);
    if (!parsed.valid) {
      $(`inference-${name}`)?.reportValidity();
      return;
    }
    if (parsed.dirty) values[name] = parsed.value;
  }
  if (!Object.keys(values).length) return;
  if (!send({inference_override: values})) {
    reportDisconnected();
    return;
  }
  pendingInferenceApply = {values, afterRevision: inferenceControlRevision};
  setInferenceControlsDisabled(true);
  updateInferenceApplyState();
}

function setInferenceControlsDisabled(disabled) {
  for (const input of $('inference-controls').querySelectorAll('input, select')) {
    input.disabled = disabled;
  }
  const button = $('apply-inference');
  if (button) button.disabled = disabled;
  if (!disabled) updateInferenceApplyState();
}

function updateInferenceControls(controls, revision = inferenceControlRevision) {
  inferenceControls = controls && typeof controls === 'object' ? controls : {};
  const entries = Object.entries(inferenceControls);
  const signature = JSON.stringify(entries.map(([name, spec]) => [
    name, spec.label, spec.description, spec.type, spec.min, spec.max, spec.step, spec.choices,
  ]));
  const container = $('inference-controls');
  if (signature !== inferenceControlSignature) {
    inferenceControlSignature = signature;
    inferenceDrafts = {};
    pendingInferenceApply = undefined;
    inferenceApplyFeedback = '';
    container.replaceChildren();
    for (const [name, spec] of entries) {
      const label = document.createElement('label');
      label.className = 'inference-control';
      label.title = spec.description || spec.label;
      label.htmlFor = `inference-${name}`;
      label.append(document.createTextNode(spec.label));
      const discrete = spec.type === 'boolean' || spec.type === 'enum';
      const input = document.createElement(discrete ? 'select' : 'input');
      input.id = `inference-${name}`;
      if (discrete) {
        for (const value of spec.type === 'boolean' ? ['true', 'false'] : spec.choices) {
          const option = document.createElement('option');
          option.value = value;
          option.textContent = value;
          input.append(option);
        }
      } else {
        input.type = 'number';
        input.min = String(spec.min);
        input.max = String(spec.max);
        input.step = spec.step == null ? 'any' : String(spec.step);
      }
      input.value = String(spec.value);
      input.defaultValue = String(spec.value);
      inferenceDrafts[name] = input.value;
      input.oninput = event => editInferenceControl(name, event.target);
      input.onkeydown = event => {
        if (event.key === 'Enter') {
          event.preventDefault();
          applyInferenceOverrides();
        }
      };
      label.append(input);
      container.append(label);
    }
    const apply = document.createElement('button');
    apply.id = 'apply-inference';
    apply.className = 'inference-apply';
    apply.type = 'button';
    apply.onclick = applyInferenceOverrides;
    apply.textContent = 'Apply settings';
    const status = document.createElement('span');
    status.id = 'inference-apply-status';
    status.className = 'inference-apply-status';
    status.setAttribute('aria-live', 'polite');
    container.append(apply, status);
  }
  container.hidden = entries.length === 0;
  const hasRevision = Number.isInteger(revision);
  const nextRevision = hasRevision ? revision : inferenceControlRevision;
  const pendingValuesApplied = pendingInferenceApply
    && Object.entries(pendingInferenceApply.values).every(
      ([name, value]) => inferenceControls[name]?.value === value,
    );
  if (pendingInferenceApply && (
    nextRevision > pendingInferenceApply.afterRevision
    || (!hasRevision && pendingValuesApplied)
  )) {
    inferenceApplyFeedback = pendingValuesApplied
      ? 'Applied atomically at the replan boundary.'
      : 'Settings were rejected; active model values were restored.';
    pendingInferenceApply = undefined;
    for (const [name, spec] of entries) inferenceDrafts[name] = String(spec.value);
  }
  inferenceControlRevision = nextRevision;
  for (const [name, spec] of entries) {
    const input = $(`inference-${name}`);
    const parsed = parsedInferenceDraft(name);
    if (input && !pendingInferenceApply && !parsed.dirty && document.activeElement !== input) {
      inferenceDrafts[name] = String(spec.value);
      input.value = String(spec.value);
      input.defaultValue = String(spec.value);
      input.dataset.dirty = 'false';
      input.setCustomValidity('');
    }
  }
  setInferenceControlsDisabled(Boolean(pendingInferenceApply));
}

function updateInference(inference) {
  if (!inference || !inference.samples) {
    $('inference-status').textContent = 'Inference: waiting for first plan';
    return;
  }
  $('inference-status').textContent = `Inference: last ${inference.last_ms.toFixed(0)} ms · rolling average ${inference.mean_ms.toFixed(0)} ms · ${inference.plans_per_second.toFixed(2)} plans/s`;
}

function updateRolloutTime(timer) {
  const seconds = timer?.elapsed_seconds;
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) {
    $('rollout-time').textContent = 'Rollout time: — (available after restart)';
    return;
  }
  const whole = Math.floor(seconds);
  const pad = value => String(value).padStart(2, '0');
  const hours = Math.floor(whole / 3600);
  const stamp = `${hours ? `${pad(hours)}:` : ''}${pad(Math.floor(whole / 60) % 60)}:${pad(whole % 60)}`;
  const pausedLabel = !timer.running && seconds > 0 ? ' · paused' : '';
  $('rollout-time').textContent = `Rollout time: ${stamp}${pausedLabel}`;
}

function chooseVelocityAction(action) {
  if (action === 'restart') {
    if (restartRollout()) $('velocity-decision').hidden = true;
    return;
  }
  if (send({velocity_action: action})) $('velocity-decision').hidden = true;
}

function configure(message) {
  ikRelaxationPending = null;
  updateIkRelaxation(message.ik_relaxation);
  updateGripperForce(message.gripper_force_limits);
  updateIkSolver(message.ik_solver);
  waitForStart = Boolean(message.wait_for_start);
  paused = Boolean(message.paused);
  started = Boolean(message.started);
  updateRolloutTime(message.rollout_time);
  videoEnabled = Boolean(message.video_recording_enabled);
  videoRecording = Boolean(message.video_recording);
  episodeEnabled = Boolean(message.episode_recording_enabled);
  episodeRequestPending = false;
  applyEpisodeState(message);
  modelBrowserEnabled = Boolean(message.model_browser_enabled);
  currentCheckpoint = message.checkpoint;
  updateCurrentModel();
  updateInferenceControls(
    message.inference_controls, message.inference_controls_revision,
  );
  overlayCamera = message.overlay_camera;
  cameras.clear();
  $('cameras').replaceChildren();
  for (const name of message.cameras) {
    const card = document.createElement('article');
    card.className = 'camera waiting';
    const head = document.createElement('div');
    head.className = 'camera-head';
    const title = document.createElement('strong');
    title.textContent = name.replaceAll('_', ' ');
    const detail = document.createElement('small');
    detail.textContent = name === overlayCamera ? 'Overlay-capable front view' : 'Live RGB';
    // A canvas, not an <img>: each rollout frame is drawn and then released.
    // Assigning a fresh data: URL per frame would leave one document resource
    // per camera per frame for the browser to hold for the whole rollout.
    const view = document.createElement('canvas');
    view.setAttribute('role', 'img');
    view.setAttribute('aria-label', `${name} live RGB`);
    const viewport = document.createElement('div');
    viewport.className = 'camera-view';
    const overlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    overlay.classList.add('reset-guide-overlay');
    overlay.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    overlay.setAttribute('aria-hidden', 'true');
    overlay.setAttribute('hidden', '');
    viewport.append(view, overlay);
    head.append(title, detail);
    card.append(head, viewport);
    $('cameras').append(card);
    cameras.set(name, {card, detail, view, image: view, overlay,
      context: view.getContext('2d'), busy: false, frameAvailable: false, imageReady: false});
  }
  renderResetGuide();
  $('overlay').checked = Boolean(message.overlay_enabled);
  $('overlay').disabled = false;
  $('start').disabled = !waitForStart || started;
  updatePauseButton();
  updateVideoControls();
  updateEpisodeControls();
  updateModelControl();
  $('restart').disabled = false;
  $('reconnect-cameras').disabled = false;
}

function frame(message) {
  updateIkRelaxation(message.ik_relaxation);
  updateGripperForce(message.gripper_force_limits);
  updateIkSolver(message.ik_solver);
  paused = Boolean(message.paused);
  started = Boolean(message.started);
  updateRolloutTime(message.rollout_time);
  videoRecording = Boolean(message.video_recording);
  applyEpisodeState(message);
  currentCheckpoint = message.checkpoint;
  updateCurrentModel();
  updateInferenceControls(
    message.inference_controls, message.inference_controls_revision,
  );
  $('start').disabled = !waitForStart || started;
  updatePauseButton();
  updateVideoControls();
  updateEpisodeControls();
  updateModelControl();
  $('status').textContent = `${message.status} · camera update ${message.age_ms} ms ago`;
  $('status').className = message.status === 'Running' ? 'running' : 'starting';
  $('overlay-status').textContent = message.overlay_enabled
    ? `${message.overlay_status} · showing on ${overlayCamera}`
    : `${message.overlay_status} · overlay hidden`;
  updateInference(message.inference);
  if ($('overlay').checked !== Boolean(message.overlay_enabled)) {
    $('overlay').checked = Boolean(message.overlay_enabled);
  }
  for (const [name, tile] of cameras) {
    tile.frameAvailable = message.images.includes(name);
    if (!tile.frameAvailable) tile.imageReady = false;
    if (message.images.includes(name)) {
      tile.card.classList.remove('waiting');
      tile.detail.textContent = name === overlayCamera && message.overlay_enabled
        ? 'Cartesian action overlay' : 'Live RGB';
    } else {
      tile.card.classList.add('waiting');
      tile.detail.textContent = 'Waiting for camera';
    }
  }
  resetGuideVisibility();
  const prompt = message.velocity_prompt;
  $('velocity-decision').hidden = !prompt;
  if (prompt) {
    $('velocity-detail').textContent = `${prompt.arms.join(' and ')} target step ${prompt.max_joint_step.toFixed(3)} rad exceeds ${prompt.limit.toFixed(3)} rad.`;
  }
}

async function drawCameraFrame(name, data) {
  const tile = cameras.get(name);
  // Drop the frame while the previous one for this camera is still decoding.
  // The next frame is 80 ms away; a backlog would never drain.
  if (!tile || tile.busy) return;
  tile.busy = true;
  try {
    const bitmap = await createImageBitmap(new Blob([data], {type: 'image/jpeg'}));
    if (tile.view.width !== bitmap.width || tile.view.height !== bitmap.height) {
      tile.view.width = bitmap.width;
      tile.view.height = bitmap.height;
    }
    tile.context.drawImage(bitmap, 0, 0);
    bitmap.close();
    tile.imageReady = true;
  } catch {
    // A truncated frame is discarded; the next one repaints the camera.
    tile.imageReady = false;
  } finally {
    tile.busy = false;
    resetGuideVisibility();
  }
}

function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds - minutes * 60);
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

function markSelectedVideo() {
  for (const row of document.querySelectorAll('.video-row')) {
    row.classList.toggle('selected', row.dataset.id === selectedVideo?.id);
  }
}

function openVideo(video) {
  selectedVideo = video;
  $('video-empty').hidden = true;
  $('video-player-wrap').hidden = false;
  $('video-title').textContent = video.id;
  $('video-meta').textContent = `${video.frames} frames · ${video.fps} fps · ${formatDuration(video.duration_seconds)} · synchronized three-camera mosaic`;
  const player = $('video-player');
  player.pause();
  player.src = `/api/videos/${encodeURIComponent(video.id)}`;
  player.load();
  markSelectedVideo();
}

async function loadVideos() {
  const list = $('video-list');
  list.textContent = 'Loading…';
  try {
    const response = await fetch('/api/videos', {cache: 'no-store'});
    if (!response.ok) throw Error(`Could not load rollout videos (${response.status})`);
    const videos = await response.json();
    list.replaceChildren();
    for (const video of videos) {
      const row = document.createElement('button');
      row.className = 'video-row';
      row.dataset.id = video.id;
      const name = document.createElement('span');
      name.className = 'video-name';
      name.textContent = video.id;
      const detail = document.createElement('span');
      detail.className = 'video-detail';
      detail.textContent = `${formatDuration(video.duration_seconds)} · ${video.frames} frames`;
      row.append(name, detail);
      row.onclick = () => openVideo(video);
      list.append(row);
    }
    if (!videos.length) {
      selectedVideo = undefined;
      $('video-player').pause();
      $('video-player').removeAttribute('src');
      $('video-player').load();
      $('video-player-wrap').hidden = true;
      $('video-empty').hidden = false;
      list.textContent = 'No completed rollout videos yet.';
      return;
    }
    const current = videos.find(video => video.id === selectedVideo?.id);
    if (current) {
      selectedVideo = current;
      markSelectedVideo();
    } else {
      openVideo(videos[0]);
    }
  } catch (error) {
    list.textContent = error.message;
  }
}

async function loadEpisodes() {
  const list = $('episode-list');
  list.textContent = 'Loading…';
  try {
    const response = await fetch('/api/episodes', {cache: 'no-store'});
    if (!response.ok) throw Error(`Could not load rollout episodes (${response.status})`);
    const episodes = await response.json();
    list.replaceChildren();
    for (const episode of episodes) {
      const row = document.createElement('div');
      row.className = `episode-row ${episode.outcome}`;
      const name = document.createElement('span');
      name.className = 'video-name';
      name.textContent = episode.filename;
      const detail = document.createElement('span');
      detail.className = 'video-detail';
      const size = (episode.size_bytes / 1e9).toFixed(1);
      detail.textContent = `${describeEpisode(episode)} · ${size} GB · ${episode.checkpoint || 'checkpoint unknown'}`;
      row.append(name, detail);
      list.append(row);
    }
    if (!episodes.length) list.textContent = 'No saved rollout episodes yet.';
  } catch (error) {
    list.textContent = error.message;
  }
}

$('stop').onclick = stopRollout;
$('start').onclick = startRollout;
$('pause').onclick = togglePause;
$('record-video').onclick = toggleVideoRecording;
$('record-episode').onclick = startEpisode;
for (const button of document.querySelectorAll('[data-episode-outcome]')) {
  button.onclick = () => saveEpisode(button.dataset.episodeOutcome);
}
$('discard-episode').onclick = discardEpisode;
$('open-episodes').onclick = () => {
  $('episodes').showModal();
  loadEpisodes();
};
$('refresh-episodes').onclick = loadEpisodes;
$('close-episodes').onclick = () => $('episodes').close();
$('select-model').onclick = () => {
  if (!modelBrowserEnabled) return;
  $('model-current').textContent = currentCheckpoint ? `Current: ${currentCheckpoint}` : 'Current checkpoint unavailable';
  $('model-search').value = '';
  $('models').showModal();
  loadModels(modelDirectory, '');
  $('model-search').focus();
};
$('open-videos').onclick = () => {
  $('videos').showModal();
  loadVideos();
};
$('refresh-videos').onclick = loadVideos;
$('close-videos').onclick = () => {
  $('video-player').pause();
  $('videos').close();
};
$('model-refresh').onclick = () => loadModels(modelDirectory);
$('model-search').oninput = () => {
  clearTimeout(modelSearchTimer);
  modelSearchTimer = setTimeout(() => loadModels(modelDirectory), 120);
};
$('model-close').onclick = () => $('models').close();
$('restart').onclick = restartRollout;
$('reconnect-cameras').onclick = reconnectCameras;
for (const button of document.querySelectorAll('[data-velocity-action]')) {
  button.onclick = () => chooseVelocityAction(button.dataset.velocityAction);
}
$('overlay').onchange = event => send({overlay: event.target.checked});
document.onkeydown = event => {
  if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
  if (
    event.target instanceof HTMLInputElement
    || event.target instanceof HTMLButtonElement
    || event.target instanceof HTMLSelectElement
    || event.target instanceof HTMLTextAreaElement
    || event.target instanceof HTMLVideoElement
  ) return;
  if (event.code === 'Space') {
    event.preventDefault();
    togglePause();
    return;
  }
  if (event.key === 'q' || event.key === 'Q' || event.key === 'Escape') {
    event.preventDefault();
    stopRollout();
  }
  if (event.key === 'c' || event.key === 'C') {
    event.preventDefault();
    startRollout();
  }
  if (event.key === 'r' || event.key === 'R') {
    event.preventDefault();
    restartRollout();
  }
  if (event.key === 'v' || event.key === 'V') {
    event.preventDefault();
    toggleVideoRecording();
  }
  if (event.key === 'd' || event.key === 'D') {
    event.preventDefault();
    toggleEpisode();
  }
  if (event.key === 'e' || event.key === 'E') {
    event.preventDefault();
    chooseVelocityAction('execute');
  }
  if (event.key === 's' || event.key === 'S') {
    event.preventDefault();
    chooseVelocityAction('resample');
  }
};

function connect() {
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    retryDelay = 500;
    ikSolverPending = null;
    ikRelaxationPending = null;
    expectedImages = [];
    $('status').textContent = 'Connected; waiting for rollout frames';
    updateInferenceApplyState();
  };
  socket.onmessage = event => {
    // Camera JPEGs arrive as binary messages, in the order the preceding frame
    // status named them.
    if (typeof event.data !== 'string') {
      const name = expectedImages.shift();
      if (name) drawCameraFrame(name, event.data);
      return;
    }
    const message = JSON.parse(event.data);
    if (message.type === 'config') configure(message);
    if (message.type === 'frame') {
      expectedImages = message.images.slice();
      frame(message);
    }
  };
  socket.onclose = event => {
    for (const tile of cameras.values()) {
      tile.frameAvailable = false;
      tile.imageReady = false;
    }
    resetGuideVisibility();
    $('ik-solver').disabled = true;
    $('ik-relaxation').disabled = true;
    $('status').className = 'starting';
    if (event.code === SUPERSEDED) {
      // A newer tab owns the dashboard. Reconnecting would only take it back
      // and leave the two tabs fighting over the rollout.
      $('status').textContent = 'A newer tab took over this dashboard; close this one.';
      for (const id of ['start', 'pause', 'record-video', 'record-episode', 'open-videos',
                        'open-episodes', 'select-model', 'restart', 'reconnect-cameras',
                        'overlay']) $(id).disabled = true;
      setInferenceControlsDisabled(true);
      return;
    }
    $('status').textContent = 'Dashboard disconnected; retrying…';
    $('start').disabled = true;
    $('pause').disabled = true;
    $('record-video').disabled = true;
    $('record-episode').disabled = true;
    $('open-videos').disabled = true;
    $('open-episodes').disabled = true;
    $('select-model').disabled = true;
    setInferenceControlsDisabled(true);
    $('restart').disabled = true;
    $('reconnect-cameras').disabled = true;
    $('overlay').disabled = true;
    // Back off: a tab left open from a finished rollout should not poll the
    // station once a second for the rest of the day.
    setTimeout(connect, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 10000);
  };
}

setupResetGuides();
connect();
