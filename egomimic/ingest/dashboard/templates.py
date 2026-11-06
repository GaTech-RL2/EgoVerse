"""The dashboard's pages, kept in code rather than separate HTML files. Names end
in .html, which turns on Jinja autoescaping: every vendor-supplied string is escaped."""

CSS = """
:root { --fg: #1d1f23; --muted: #626873; --line: #dde0e5; --bg: #ffffff;
  --panel: #f6f7f9; --accent: #2457c5; --ok: #1b7a3d; --warn: #9a5b00; --bad: #b42318; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e8eb; --muted: #9aa1ab; --line: #343a42; --bg: #15181c;
    --panel: #1d2126; --accent: #7aa2ff; --ok: #5cc98a; --warn: #e3a54a; --bad: #ff7b70; }
}
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.5 system-ui, sans-serif; color: var(--fg);
  background: var(--bg); }
header { display: flex; align-items: center; gap: 16px; padding: 12px 16px;
  border-bottom: 1px solid var(--line); }
header .who { margin-left: auto; color: var(--muted); }
header form { display: inline; }
main { max-width: 1100px; margin: 0 auto; padding: 16px; }
a { color: var(--accent); }
h1 { font-size: 20px; margin: 8px 0 16px; }
h2 { font-size: 16px; margin: 24px 0 8px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { color: var(--muted); font-weight: 500; }
.scroll { overflow-x: auto; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); }
.state { white-space: nowrap; font-weight: 500; }
.state.committed, .state.ready, .state.approved { color: var(--ok); }
.state.rejected, .state.failed, .state.unsupported, .state.declined { color: var(--bad); }
.state.uploading, .state.queued, .state.converting,
.state.commit_rejected { color: var(--warn); }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
  padding: 12px 16px; }
ul.problems { margin: 0; padding-left: 20px; }
pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
input, select, button { font: inherit; padding: 6px 8px; border: 1px solid var(--line);
  border-radius: 4px; background: var(--bg); color: var(--fg); }
button { cursor: pointer; }
.login { max-width: 360px; margin: 64px auto; display: grid; gap: 12px; }
.error { color: var(--bad); }
.flash { padding: 8px 12px; border-radius: 4px; border: 1px solid var(--line); }
.flash.ok { color: var(--ok); }
.review { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 12px; }
.review input[name=comment] { flex: 1 1 280px; }
.upload { display: grid; gap: 12px; max-width: 720px; }
.button { display: inline-block; padding: 6px 12px; border: 1px solid var(--accent);
  border-radius: 4px; text-decoration: none; }
progress { width: 100%; }
/* Shared visual language: quiet surfaces, clear hierarchy, visible focus. */
:root { --bg: #f5f7fa; --panel: #fff; --fg: #182b36; --muted: #586b78;
  --line: #dce4e9; --accent: #176c62; --ok: #176c62; --warn: #8b5c16;
  --bad: #ad3542; --soft: #edf3f5; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #111b22; --panel: #19262f; --fg: #e4edf2; --muted: #a4b6c2;
    --line: #34454f; --accent: #81d5be; --ok: #81d5be; --warn: #edc078;
    --bad: #ffa3ad; --soft: #22333e; }
}
body { line-height: 1.6; }
header { background: var(--panel); padding: 16px max(24px, calc((100vw - 1200px) / 2));
  flex-wrap: wrap; gap: 20px; }
.brand { display: flex; align-items: center; gap: 10px; font-size: 18px;
  letter-spacing: -.5px; text-decoration: none; color: var(--fg); font-weight: 750; }
.brand-mark { width: 30px; height: 30px; display: grid; place-items: center;
  border-radius: 9px; background: var(--accent); color: var(--panel); font-size: 16px; }
header nav { display: flex; gap: 20px; }
header nav a { color: var(--muted); text-decoration: none; font-weight: 550; }
header nav a[aria-current=page] { color: var(--accent); }
header .who { font-size: 13px; }
main { max-width: 1248px; padding: 40px 24px 72px; }
h1 { font-size: clamp(25px, 4vw, 34px); line-height: 1.2; letter-spacing: -1px;
  margin: 6px 0 12px; overflow-wrap: anywhere; }
h2 { font-size: 19px; letter-spacing: -.3px; margin-top: 32px; }
.eyebrow { color: var(--accent); text-transform: uppercase; font-size: 11px;
  font-weight: 750; letter-spacing: 1.6px; margin: 0 0 8px; }
.page-head { display: flex; align-items: center; justify-content: space-between;
  gap: 20px; margin-bottom: 28px; }
.page-head p { margin: 0; }
.summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px; margin-bottom: 24px; }
.metric { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 18px 22px; }
.metric strong { display: block; font-size: 28px; line-height: 1.3; font-weight: 650; }
.metric span { color: var(--muted); font-size: 13px; }
.panel, .scroll, .login, .upload { border: 1px solid var(--line); border-radius: 12px;
  background: var(--panel); }
.panel { padding: 20px 24px; overflow-wrap: anywhere; }
.scroll { overflow-x: auto; }
th, td { padding: 14px 16px; }
th { font-size: 12px; background: var(--soft); }
td { font-size: 14px; overflow-wrap: anywhere; }
tr:last-child td { border-bottom: 0; }
tr:hover td { background: var(--soft); }
td a { font-weight: 600; text-decoration: none; }
.state { display: inline-block; border: 1px solid currentColor; border-radius: 6px;
  padding: 2px 8px; font-size: 12px; font-weight: 650; }
.filters { gap: 10px; margin-bottom: 18px; align-items: center; }
input, select, button, .button { border-radius: 7px; padding: 9px 12px; }
button, .button { font-weight: 600; }
button:hover, .button:hover { border-color: var(--accent); }
button:disabled { cursor: wait; opacity: .6; }
button[value=approve], .primary, .login button, #go { background: var(--accent);
  border-color: var(--accent); color: var(--panel); }
button[value=decline] { color: var(--bad); }
input[type=checkbox] { width: 17px; height: 17px; accent-color: var(--accent); }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 3px; }
.review { padding: 16px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; }
.login { padding: 32px; max-width: 420px; margin: 48px auto; gap: 18px; }
.login input { width: 100%; margin-top: 5px; }
.upload { max-width: 850px; padding: 24px; gap: 20px; }
.upload input[type=file] { display: block; width: 100%; margin-top: 8px;
  padding: 18px; border-style: dashed; }
.upload label { font-weight: 600; }
.upload .panel { background: var(--soft); }
.empty { text-align: center; padding: 48px 24px; }
.empty h2 { margin: 0 0 8px; }
.flash { background: var(--panel); padding: 14px 18px; }
code { font-size: .88em; background: var(--soft); padding: 2px 5px; border-radius: 4px;
  overflow-wrap: anywhere; }
.reference { display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 36px; }
.reference nav { position: sticky; top: 24px; align-self: start; display: grid; gap: 8px; }
.reference nav a { text-decoration: none; color: var(--muted); font-size: 14px; }
.reference section { scroll-margin-top: 24px; margin-bottom: 28px; }
.reference section > h2 { margin-top: 0; }
.flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px; padding: 0; list-style: none; margin: 0 0 28px; counter-reset: step; }
.flow li { padding: 16px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; counter-increment: step; }
.flow li::before { content: '0' counter(step); display: block; color: var(--accent);
  font-size: 12px; font-weight: 700; margin-bottom: 8px; }
.flow small { display: block; color: var(--muted); }
@media (max-width: 700px) {
  header { padding: 14px 16px; gap: 12px; }
  header .who { margin-left: 0; }
  header nav { gap: 12px; }
  main { padding: 24px 16px 48px; }
  .page-head { align-items: flex-start; flex-direction: column; }
  .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .metric { padding: 14px; }
  .reference { grid-template-columns: 1fr; gap: 24px; }
  .reference nav { position: static; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .flow { grid-template-columns: 1fr; }
  .login { margin: 20px auto; padding: 24px; }
  .filters select { max-width: 100%; }
  .upload { padding: 18px; }
}
"""

BASE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block title %}EgoVerse deliveries{% endblock %}</title>
<link rel="stylesheet" href="{{ url_for('style') }}"></head>
<body>
<header><a class="brand" href="{{ url_for('index') }}"><span class="brand-mark" aria-hidden="true">E</span>EgoVerse <span class="muted">/ deliveries</span></a>
{% if user %}<nav aria-label="Main navigation"><a href="{{ url_for('index') }}" {{ 'aria-current=page' if request.endpoint == 'index' }}>All deliveries</a>
</nav>
<span class="who">{{ user }} · {{ "staff" if staff else account.vendor }}</span>
<form method="post" action="{{ url_for('logout') }}">
<input type="hidden" name="csrf" value="{{ csrf }}"><button>Log out</button></form>
{% endif %}</header>
<main>{% for kind, text in get_flashed_messages(with_categories=true) %}
<p class="flash {{ kind }}" role="status">{{ text }}</p>{% endfor %}
{% block body %}{% endblock %}</main>
</body></html>"""

LOGIN = """{% extends "base.html" %}
{% block title %}Log in · EgoVerse deliveries{% endblock %}
{% block body %}
<form class="login" method="post">
<div><p class="eyebrow">Delivery workspace</p>
<h1>Log in</h1>
<p class="muted">Upload recordings, follow conversion, and review what enters EgoVerse.</p></div>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
<input type="hidden" name="csrf" value="{{ csrf }}">
<label>User<br><input name="user" autocomplete="username" required autofocus></label>
<label>Password<br><input name="password" type="password"
  autocomplete="current-password" required></label>
<button>Log in</button>
</form>
{% endblock %}"""

INDEX = """{% extends "base.html" %}
{% block body %}
<div class="page-head"><div><p class="eyebrow">{{ 'Review workspace' if staff else 'Vendor workspace' }}</p>
<h1>{{ vendor or "All vendors" }}: deliveries</h1>
<p class="muted">{{ 'Follow incoming recordings from upload to reviewed dataset.' if staff else 'Send recordings and track their progress into the dataset.' }}</p></div>
{% if not staff %}<a class="button primary" href="{{ url_for('new') }}">New delivery</a>{% endif %}</div>
<div class="summary" aria-label="Delivery summary for the selected vendor">
{% for label, value in summary %}<div class="metric"><strong>{{ value }}</strong><span>{{ label }}</span></div>{% endfor %}
</div>
<p class="muted">Summary covers all states{{ ' for ' ~ vendor if vendor else ' across all vendors' }}; filters below narrow the table.</p>
<form class="filters" method="get">
{% if staff %}<select name="vendor" aria-label="Filter by vendor"><option value="">All vendors</option>
{% for v in vendors %}<option {{ "selected" if v == vendor }}>{{ v }}</option>{% endfor %}
</select>{% endif %}
<select name="state" aria-label="Filter by state"><option value="">Any state</option>
{% for s, words in STATES.items() %}
<option value="{{ s }}" {{ "selected" if s == state }}>{{ words }}</option>{% endfor %}
</select><button>Filter</button></form>
{% if not rows %}<div class="panel empty"><h2>{{ 'No matching deliveries' if state else 'No deliveries yet.' }}</h2>
<p class="muted">{{ 'Choose another state to see more recordings.' if state else 'Deliveries will appear here once files arrive.' }}</p></div>{% else %}
{% if staff %}<form method="post" action="{{ url_for('review.decide_many') }}">
<input type="hidden" name="csrf" value="{{ csrf }}">
<input type="hidden" name="back" value="{{ request.full_path }}">{% endif %}
<div class="scroll"><table>
<tr>{% if staff %}<th><input type="checkbox" id="all" title="Select every one waiting"
  aria-label="Select all waiting for review"></th><th>Vendor</th>{% endif %}<th>Delivery</th><th>State</th>
<th class="num">Size</th><th class="num">Frames</th><th class="num">Problems</th>
<th>Updated (UTC)</th></tr>
{% for r in rows %}<tr>
{% if staff %}<td>{% if r.decidable %}<input type="checkbox" name="item" class="pick"
  value="{{ r.ref.vendor }}/{{ r.ref.delivery_id }}/{{ r.status.stage }}"
  aria-label="Select {{ r.ref.delivery_id }}">{% endif %}</td>
<td>{{ r.ref.vendor }}</td>{% endif %}
<td><a href="{{ url_for('delivery', vendor=r.ref.vendor,
  delivery_id=r.ref.delivery_id) }}">{{ r.ref.delivery_id }}</a></td>
<td><span class="state {{ r.state }}" title="{{ STATES.get(r.state, '') }}">{{ r.state }}</span></td>
<td class="num">{{ r.size | size }}</td>
<td class="num">{{ r.frames if r.frames is not none else "" }}</td>
<td class="num">{{ r.problems | length or "" }}</td>
<td class="muted">{{ r.updated | when }}</td>
</tr>{% endfor %}
</table></div>
{% if staff %}<div class="review">
<input name="comment" aria-label="Review comment" maxlength="1000" placeholder="Comment (needed to decline; vendors see it)">
<button name="decision" value="approve">Approve selected</button>
<button name="decision" value="decline">Decline selected</button>
</div></form><script src="{{ url_for('review_js') }}"></script>{% endif %}
{% endif %}
{% endblock %}"""

DELIVERY = """{% extends "base.html" %}
{% block title %}{{ r.ref.delivery_id }} · EgoVerse deliveries{% endblock %}
{% block body %}
<p class="eyebrow">Delivery details</p>
<h1>{{ r.ref.vendor }} / {{ r.ref.delivery_id }}</h1>
{% if not staff %}<p><a class="button" href="{{ url_for('new', delivery=r.ref.delivery_id) }}">
Upload a correction</a></p>{% endif %}
<div class="panel"><span class="state {{ r.state }}">{{ r.state }}</span>:
{{ STATES.get(r.state, "") }}
{% if r.status.recording_id %}<br><span class="muted">Recording</span>
{{ r.status.recording_id }}{% endif %}
{% if r.status.format %}<br><span class="muted">Format</span>
{{ r.status.format }}{% endif %}
{% if decision %}<br><span class="muted">Decision</span> {{ decision.decision }}
{% if decision.comment %}: {{ decision.comment }}{% endif %}
{% if decision.who %}<span class="muted">by {{ decision.who }}</span>{% endif %}
{% endif %}</div>

{% if staff and r.decidable %}<h2>Review</h2>
<form class="panel review" method="post"
  action="{{ url_for('review.decide_one', vendor=r.ref.vendor, delivery_id=r.ref.delivery_id) }}">
<input type="hidden" name="csrf" value="{{ csrf }}">
<input type="hidden" name="stage" value="{{ r.status.stage }}">
<span class="muted">Result {{ r.status.stage[:12] }}{% if r.decision %}: currently
<strong>{{ r.decision.decision }}d</strong> by {{ r.decision.who }}{% endif %}</span>
<input name="comment" aria-label="Review comment" maxlength="1000" placeholder="Comment (needed to decline; the vendor sees it)">
<button name="decision" value="approve">Approve</button>
<button name="decision" value="decline">Decline</button>
</form>{% endif %}

{% if r.problems %}<h2>Problems ({{ r.problems | length }})</h2>
<div class="panel"><ul class="problems">
{% for p in r.problems %}<li>{{ p }}</li>{% endfor %}</ul></div>{% endif %}

{% if error %}<h2>Error</h2><div class="panel"><pre>{{ error }}</pre></div>{% endif %}

{% if r.status.report %}<h2>Conversion</h2><div class="scroll"><table>
{% for k, v in r.status.report.items() %}<tr><th>{{ k }}</th><td>
{% if v is mapping %}{% for kk, vv in v.items() %}{{ kk }}: {{ vv | fmt }}{{ ", " if not loop.last }}
{% endfor %}{% else %}{{ v | fmt }}{% endif %}</td></tr>{% endfor %}
</table></div>{% endif %}

{% if r.status.diff %}<h2>What changes in the dataset</h2><div class="scroll"><table>
{% for k, v in r.status.diff.items() %}<tr><th>{{ k }}</th><td>
{% if v is mapping %}{% for kk, vv in v.items() %}{{ kk }}: {{ vv | fmt }}<br>
{% endfor %}{% else %}{{ v | fmt }}{% endif %}</td></tr>{% endfor %}
</table></div>{% endif %}

<h2>Files</h2>{% if r.files %}<div class="scroll"><table>
<tr><th>Name</th><th class="num">Size</th></tr>
{% for name, (size, etag) in r.files | dictsort %}
<tr><td>{{ name }}</td><td class="num">{{ size | size }}</td></tr>{% endfor %}
</table></div>{% else %}<p class="muted">No files.</p>{% endif %}

{% if staff and r.status.provenance %}<h2>Made by</h2><div class="scroll"><table>
{% for k, v in r.status.provenance.items() %}<tr><th>{{ k }}</th><td>{{ v | fmt }}</td></tr>
{% endfor %}</table></div>{% endif %}

{% if r.status.history %}<h2>History</h2><div class="scroll"><table>
{% for h in r.status.history | reverse %}
<tr><td><span class="state {{ h.state }}">{{ h.state }}</span></td>
<td class="muted">{{ h.at | when }}</td></tr>{% endfor %}
</table></div>{% endif %}

{% if staff and decisions %}<h2>Decisions</h2><div class="scroll"><table>
<tr><th>Decision</th><th>By</th><th>Comment</th><th>Stage</th><th>At (UTC)</th></tr>
{% for d in decisions | reverse %}<tr><td>{{ d.decision }}</td><td>{{ d.who }}</td>
<td>{{ d.comment }}</td><td class="muted">{{ (d.stage or "")[:12] }}</td>
<td class="muted">{{ d.at | when }}</td></tr>{% endfor %}
</table></div>{% endif %}
{% endblock %}"""

NEW = """{% extends "base.html" %}
{% block title %}New delivery · EgoVerse deliveries{% endblock %}
{% block body %}
<h1>{{ "Correct " + delivery_id if delivery_id else "New delivery" }}</h1>
<form id="upload" class="upload" data-csrf="{{ csrf }}">
<label>Delivery name<br><input name="delivery" value="{{ delivery_id }}" required
  pattern="[A-Za-z0-9][A-Za-z0-9._\\-]{0,127}" {{ "readonly" if delivery_id }}
  placeholder="e.g. fold_clothes_scene7_rec4"></label>
<p class="muted">Letters, digits, <code>.</code> <code>_</code> <code>-</code>; one
recording per delivery. Uploading to an existing name replaces its files: send the whole
delivery again, not only what changed.</p>
<label>Folder<br><input name="folder" type="file" webkitdirectory multiple></label>
<label>or files<br><input name="files" type="file" multiple></label>
<div class="panel"><strong>Accepted</strong>
<ul class="problems">{% for f in formats %}<li><strong>{{ f.name }}</strong>:
{{ f.required | sort | join(", ") }}{% if f.optional %} (optional:
{{ f.optional | sort | join(", ") }}){% endif %}. {{ f.description }}</li>{% endfor %}
</ul><span class="muted">At most {{ limit_gib }} GiB per file.</span></div>
<table id="picked" class="scroll"></table>
<button id="go">Upload</button>
<p class="muted">Large deliveries or scripts: the
<a href="{{ url_for('upload_cli') }}">command-line uploader</a> (Python 3.10+, nothing
to install): <code>python egoverse_upload.py --url {{ request.host_url.rstrip("/") }}
--user {{ user }} &lt;folder&gt;</code></p>
<progress id="bar" max="1" value="0" hidden></progress>
<p id="msg" role="status"></p>
</form>
<script src="{{ url_for('upload_js') }}"></script>
{% endblock %}"""

UPLOAD_JS = """'use strict';
// Upload a delivery: each file goes straight to the bucket in equal parts through
// presigned links (the dashboard never sees the bytes), then the delivery is marked
// complete as exactly these files. See egomimic/ingest/dashboard/uploads.py.
(() => {
  const form = document.getElementById('upload');
  if (!form) return;
  const csrf = form.dataset.csrf;
  const $ = (id) => document.getElementById(id);
  const PARALLEL = 4, RETRIES = 4, BATCH = 100;
  let busy = false;

  const say = (text, bad) => { $('msg').textContent = text; $('msg').className = bad ? 'error' : ''; };
  const human = (n) => { const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0;
    while (n >= 1000 && i < u.length - 1) { n /= 1000; i++; } return (i ? n.toFixed(1) : n) + ' ' + u[i]; };

  async function api(method, path, body) {
    const r = await fetch(path, { method, credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrf },
      body: body === undefined ? undefined : JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) { const e = new Error(data.error || r.status + ' ' + r.statusText); e.data = data; throw e; }
    return data;
  }

  // Relative names: drop the picked folder's own name, keep subfolders.
  function picked() {
    const folder = [...form.folder.files], files = [...form.files.files];
    const list = folder.length ? folder : files;
    return list.map((f) => ({ file: f,
      name: f.webkitRelativePath ? f.webkitRelativePath.split('/').slice(1).join('/') : f.name }));
  }

  function show() {
    const t = $('picked'); t.replaceChildren();
    for (const { file, name } of picked()) {
      const tr = t.insertRow(); tr.insertCell().textContent = name;
      const td = tr.insertCell(); td.className = 'num'; td.textContent = human(file.size);
    }
  }
  form.folder.addEventListener('change', show);
  form.files.addEventListener('change', show);

  async function retry(fn) {
    for (let i = 0; ; i++) {
      try { return await fn(); } catch (e) {
        if (i + 1 >= RETRIES) throw e;
        await new Promise((ok) => setTimeout(ok, 1000 * 2 ** i));
      }
    }
  }

  async function putPart(base, name, start, n, url, blob) {
    if (url) {  // presigned: straight to the bucket; its CORS must expose ETag
      const r = await fetch(url, { method: 'PUT', body: blob });
      if (!r.ok) throw new Error('part ' + n + ': ' + r.status);
      const etag = r.headers.get('ETag');
      if (!etag) throw new Error('the bucket does not expose ETag (CORS)');
      return etag;
    }
    const q = new URLSearchParams({ name, upload_id: start.upload_id, part: n });
    const r = await fetch(base + '/part?' + q, { method: 'PUT', body: blob,
      credentials: 'same-origin', headers: { 'X-CSRF-Token': csrf } });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || 'part ' + n + ': ' + r.status);
    return data.etag;
  }

  async function uploadFile(base, { file, name }, progress) {
    const start = await api('POST', base + '/uploads', { name, size: file.size });
    const ref = { name, upload_id: start.upload_id };
    try {
      const etags = [];
      for (let first = 1; first <= start.parts; first += BATCH) {
        const nums = [];
        for (let n = first; n < Math.min(first + BATCH, start.parts + 1); n++) nums.push(n);
        const { urls } = await api('POST', base + '/urls', { ...ref, parts: nums });
        const queue = [...nums];
        const worker = async () => {
          for (let n; (n = queue.shift()) !== undefined;) {
            const blob = file.slice((n - 1) * start.part_size, n * start.part_size);
            etags.push([n, await retry(() => putPart(base, name, start, n, urls[n], blob))]);
            progress(blob.size);
          }
        };
        await Promise.all(Array.from({ length: PARALLEL }, worker));
      }
      await api('POST', base + '/finish', { ...ref, parts: etags.sort((a, b) => a[0] - b[0]) });
    } catch (e) {
      await api('POST', base + '/abort', ref).catch(() => {});
      throw e;
    }
  }

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (busy) return;
    const id = form.delivery.value.trim(), files = picked();
    if (!files.length) return say('Pick a folder or files first.', true);
    const base = '/api/deliveries/' + encodeURIComponent(id);
    busy = true; $('go').disabled = true;
    const total = files.reduce((n, f) => n + f.file.size, 0) || 1;
    let done = 0;
    const bar = $('bar'); bar.hidden = false; bar.value = 0;
    window.onbeforeunload = () => 'The upload is still running.';
    try {
      const now = await api('GET', base);
      if (now.uploads.length) {
        if (!confirm(now.uploads.length + ' unfinished upload(s) to ' + id +
            ' (e.g. a closed tab). Drop them and continue?')) throw new Error('Stopped.');
        for (const u of now.uploads) await api('POST', base + '/clear', { record: u.record });
      }
      for (const [i, f] of files.entries()) {
        say('Uploading ' + f.name + ' (' + (i + 1) + ' of ' + files.length + ')');
        await uploadFile(base, f, (n) => { done += n; bar.value = done / total; });
      }
      say('Finishing...');
      await api('POST', base + '/complete', { files: files.map((f) => f.name) });
      window.onbeforeunload = null;
      location.href = '/d/' + now.vendor + '/' + encodeURIComponent(id);
    } catch (e) {
      say('Upload failed: ' + e.message, true);
    } finally {
      busy = false; $('go').disabled = false; window.onbeforeunload = null;
    }
  });
})();
"""

REVIEW_JS = """'use strict';
// Staff review: the header checkbox ticks every delivery waiting for review.
(() => {
  const all = document.getElementById('all');
  if (!all) return;
  const picks = () => [...document.querySelectorAll('input.pick')];
  all.addEventListener('change', () => picks().forEach((c) => { c.checked = all.checked; }));
})();
"""

ERROR = """{% extends "base.html" %}
{% block title %}{{ name }} · EgoVerse deliveries{% endblock %}
{% block body %}
<h1>{{ name }}</h1>
<p>{{ text }}</p>
<p><a href="{{ url_for('index') if user else url_for('login') }}">Back to
{{ "the deliveries" if user else "log in" }}</a></p>
{% endblock %}"""


TEMPLATES = {
    "base.html": BASE,
    "login.html": LOGIN,
    "index.html": INDEX,
    "delivery.html": DELIVERY,
    "new.html": NEW,
    "error.html": ERROR,
}


def size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return ""


def when(at: str) -> str:
    """store.now() stamps, 20261021T140312.123456Z -> 2026-10-21 14:03"""
    if len(at) < 13:
        return at or ""
    return f"{at[:4]}-{at[4:6]}-{at[6:8]} {at[9:11]}:{at[11:13]}"


def fmt(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {fmt(x)}" for k, x in v.items()) if v else "none"
    if isinstance(v, (list, tuple)):
        return ", ".join(fmt(x) for x in v) if v else "none"
    return str(v)


FILTERS = {"size": size, "when": when, "fmt": fmt}
