"""
api.py - FastAPI backend + HTML dashboard for local benchmark observability.

Run:
    python -m kvmemory.dashboard.api --obs-dir .kvmem_obs

Then open http://localhost:8000
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..observability import ObservabilityStore
from ..storage.vector_db import VectorDB


def _parse_time(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value).timestamp()


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>KV Memory Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e2e8f0; --muted: #8892a4; --accent: #6366f1;
    --green: #22c55e; --yellow: #eab308; --red: #ef4444; --blue: #3b82f6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; letter-spacing: -0.3px; }
  header .badge { background: var(--accent); color: #fff; font-size: 11px; padding: 2px 8px; border-radius: 999px; }
  .layout { display: grid; grid-template-columns: 300px 1fr; min-height: calc(100vh - 49px); }
  .sidebar { background: var(--surface); border-right: 1px solid var(--border); padding: 16px; overflow-y: auto; }
  .sidebar h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 10px; }
  .run-card { border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 8px; cursor: pointer; transition: border-color .15s; }
  .run-card:hover, .run-card.active { border-color: var(--accent); }
  .run-card .run-id { font-size: 11px; color: var(--muted); font-family: monospace; }
  .run-card .run-model { font-weight: 600; font-size: 13px; margin: 4px 0 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .run-card .run-meta { font-size: 11px; color: var(--muted); }
  .status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
  .status-completed { background: var(--green); }
  .status-running { background: var(--yellow); animation: pulse 1.2s infinite; }
  .status-failed { background: var(--red); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .main { padding: 24px; overflow-y: auto; }
  .placeholder { display: flex; align-items: center; justify-content: center; height: 300px; color: var(--muted); font-size: 15px; }
  .section { margin-bottom: 28px; }
  .section-title { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; }
  .metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .metric-label { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
  .metric-value { font-size: 22px; font-weight: 700; }
  .metric-sub { font-size: 11px; color: var(--muted); margin-top: 4px; }
  .green { color: var(--green); } .yellow { color: var(--yellow); } .blue { color: var(--blue); } .red { color: var(--red); }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); padding: 8px 12px; border-bottom: 1px solid var(--border); }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
  tr:last-child td { border-bottom: none; }
  .bar-row { display: flex; align-items: center; gap: 8px; }
  .bar-bg { flex: 1; background: var(--border); border-radius: 3px; height: 8px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width .4s ease; }
  .tag { display: inline-block; background: var(--border); border-radius: 4px; padding: 1px 6px; font-size: 11px; font-family: monospace; }
  .config-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
  .config-item { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
  .config-key { font-size: 11px; color: var(--muted); margin-bottom: 3px; }
  .config-val { font-family: monospace; font-size: 12px; }
  #refresh-btn { margin-left: auto; background: var(--border); border: none; color: var(--text); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; }
  #refresh-btn:hover { background: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>KV Memory Dashboard</h1>
  <span class="badge">Local</span>
  <button id="refresh-btn" onclick="loadRuns()">⟳ Refresh</button>
</header>
<div class="layout">
  <div class="sidebar">
    <h2>Runs</h2>
    <div id="run-list"><div style="color:var(--muted);font-size:12px;">Loading…</div></div>
  </div>
  <div class="main" id="main-panel">
    <div class="placeholder">← Select a run to view results</div>
  </div>
</div>

<script>
let runs = [];
let selectedId = null;

function fmt(n, decimals=1) {
  if (n == null) return '—';
  return Number(n).toFixed(decimals);
}
function pct(n) { return n == null ? '—' : fmt(n*100,1) + '%'; }
function ms(n)  { return n == null ? '—' : fmt(n,0) + ' ms'; }
function trunc(s, n=40) { return s && s.length > n ? s.slice(0,n)+'…' : (s||'—'); }

function statusClass(s) {
  if (s === 'completed') return 'status-completed';
  if (s === 'running')   return 'status-running';
  return 'status-failed';
}

async function loadRuns() {
  const res = await fetch('/api/runs');
  const data = await res.json();
  runs = data.runs || [];
  const list = document.getElementById('run-list');
  if (!runs.length) { list.innerHTML = '<div style="color:var(--muted);font-size:12px;">No runs found.</div>'; return; }
  list.innerHTML = runs.map(r => {
    const model = (r.config?.model || r.config?.model_id || '').replace(/.*[\\/]/, '');
    const f1 = r.summary?.kv_metrics?.overall_f1;
    const f1str = f1 != null ? ` · F1 ${pct(f1)}` : '';
    const d = new Date((r.started_at||0)*1000);
    const ts = d.toLocaleString(undefined, {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
    return `<div class="run-card${r.run_id===selectedId?' active':''}" onclick="selectRun('${r.run_id}')">
      <div class="run-id"><span class="status-dot ${statusClass(r.status)}"></span>${r.run_id}</div>
      <div class="run-model" title="${r.config?.model||''}">${model||'unknown'}</div>
      <div class="run-meta">${ts}${f1str} · n=${r.metadata?.question_count??'?'}</div>
    </div>`;
  }).join('');
}

async function selectRun(id) {
  selectedId = id;
  // re-render sidebar highlight
  document.querySelectorAll('.run-card').forEach(c => {
    c.classList.toggle('active', c.onclick?.toString().includes(id));
  });
  const res = await fetch(`/api/runs/${id}`);
  const run = await res.json();
  renderRun(run);
}

function metricCard(label, value, cls='', sub='') {
  return `<div class="metric-card">
    <div class="metric-label">${label}</div>
    <div class="metric-value ${cls}">${value}</div>
    ${sub ? `<div class="metric-sub">${sub}</div>` : ''}
  </div>`;
}

function barRow(val, max, cls) {
  const pct = max > 0 ? Math.min(100, (val/max)*100) : 0;
  return `<div class="bar-row">
    <span style="width:56px;text-align:right">${fmt(val,0)}</span>
    <div class="bar-bg"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
  </div>`;
}

function renderRun(run) {
  const kv  = run.summary?.kv_metrics || {};
  const rag = run.summary?.lexical_rag_metrics || {};
  const sw  = run.summary?.sliding_window_metrics || {};
  const cfg = run.config || {};
  const meta = run.metadata || {};

  // Accuracy comparison table rows by type
  const allTypes = new Set([
    ...Object.keys(kv.f1_by_type||{}),
    ...Object.keys(rag.f1_by_type||{}),
    ...Object.keys(sw.f1_by_type||{}),
  ]);
  const typeRows = [...allTypes].map(t => `
    <tr>
      <td><span class="tag">${t.replace(/_/g,' ')}</span></td>
      <td class="green">${pct(kv.f1_by_type?.[t])}</td>
      <td class="yellow">${pct(rag.f1_by_type?.[t])}</td>
      <td class="blue">${pct(sw.f1_by_type?.[t])}</td>
    </tr>`).join('');

  // Latency bar max
  const maxLat = Math.max(kv.avg_latency_ms||0, rag.avg_latency_ms||0, sw.avg_latency_ms||0, 1);
  const maxTok = Math.max(kv.avg_prefill_tokens||0, rag.avg_prefill_tokens||0, sw.avg_prefill_tokens||0, 1);

  // Config items
  const cfgItems = Object.entries(cfg).map(([k,v]) =>
    `<div class="config-item"><div class="config-key">${k}</div><div class="config-val">${JSON.stringify(v)}</div></div>`
  ).join('');

  // Per-question answers table
  const kvRes  = run.summary?.kv_results || [];
  const ragRes = run.summary?.lexical_rag_results || [];
  const swRes  = run.summary?.sliding_window_results || [];
  // Build lookup by question_id for rag/sw
  const ragMap = Object.fromEntries(ragRes.map(r => [r.question_id, r]));
  const swMap  = Object.fromEntries(swRes.map(r => [r.question_id, r]));

  const answerRows = kvRes.map(r => {
    const rag_r = ragMap[r.question_id] || {};
    const sw_r  = swMap[r.question_id]  || {};
    const f1cls = r.f1_score > 0.5 ? 'green' : r.f1_score > 0.2 ? 'yellow' : 'red';
    return `<tr>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${(r.question||'').replace(/"/g,'&quot;')}">${r.question_type?.replace(/_/g,' ')||''}<br><span style="color:var(--muted);font-size:11px">${(r.question||'').slice(0,60)}…</span></td>
      <td style="color:var(--muted);font-size:11px">${(r.gold_answer||'').slice(0,80)}</td>
      <td class="${f1cls}" style="max-width:200px" title="${(r.predicted_answer||'').replace(/"/g,'&quot;')}">${(r.predicted_answer||'').slice(0,80)}</td>
      <td class="${f1cls}">${fmt(r.f1_score*100,0)}%</td>
      <td style="color:var(--muted)">${(rag_r.predicted_answer||'—').slice(0,60)}</td>
      <td style="color:var(--muted)">${fmt((rag_r.f1_score||0)*100,0)}%</td>
    </tr>`;
  }).join('');

  const answersSection = kvRes.length ? `
    <div class="section">
      <div class="section-title">Per-Question Answers</div>
      <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>Question</th>
          <th>Gold Answer</th>
          <th class="green">KV Predicted</th>
          <th class="green">KV F1</th>
          <th style="color:var(--muted)">RAG Predicted</th>
          <th style="color:var(--muted)">RAG F1</th>
        </tr></thead>
        <tbody>${answerRows}</tbody>
      </table>
      </div>
    </div>` : '';

  const kvBreakdown = (kv.avg_stage1_ms != null) ? `
    <div class="section">
      <div class="section-title">KV Memory Latency Breakdown</div>
      <table>
        <thead><tr><th>Stage</th><th>Avg (ms)</th></tr></thead>
        <tbody>
          <tr><td>Stage 1 ANN</td><td>${ms(kv.avg_stage1_ms)}</td></tr>
          <tr><td>Stage 2 MMR</td><td>${ms(kv.avg_stage2_ms)}</td></tr>
          <tr><td>Fetch KV</td><td>${ms(kv.avg_fetch_ms)}</td></tr>
          <tr><td>Inject + Generate</td><td>${ms(kv.avg_generate_ms)}</td></tr>
        </tbody>
      </table>
    </div>` : '';

  const reduction = kv.avg_prefill_tokens != null && rag.avg_prefill_tokens > 0
    ? (1 - kv.avg_prefill_tokens / rag.avg_prefill_tokens) * 100 : null;

  document.getElementById('main-panel').innerHTML = `
    <div class="section">
      <div class="section-title">Run · ${run.run_id}</div>
      <div class="metrics-grid">
        ${metricCard('KV F1', pct(kv.overall_f1), 'green')}
        ${metricCard('RAG F1', pct(rag.overall_f1), 'yellow')}
        ${metricCard('Sliding Win F1', pct(sw.overall_f1), 'blue')}
        ${metricCard('Prefill Reduction', reduction!=null ? fmt(reduction,1)+'%' : '—', 'green', 'KV vs RAG')}
        ${metricCard('KV Avg Latency', ms(kv.avg_latency_ms), '')}
        ${metricCard('Questions', meta.question_count ?? run.summary?.question_count ?? '—')}
      </div>
    </div>

    <div class="section">
      <div class="section-title">F1 by Question Type</div>
      <table>
        <thead><tr><th>Type</th><th class="green">KV Memory</th><th class="yellow">RAG</th><th class="blue">Sliding Win</th></tr></thead>
        <tbody>${typeRows || '<tr><td colspan="4" style="color:var(--muted)">No per-type data</td></tr>'}</tbody>
      </table>
    </div>

    <div class="section">
      <div class="section-title">Avg Total Latency (ms)</div>
      <table>
        <thead><tr><th>Method</th><th>Latency</th></tr></thead>
        <tbody>
          <tr><td><span class="green">KV Memory</span></td><td>${barRow(kv.avg_latency_ms||0, maxLat, 'green')}</td></tr>
          <tr><td><span class="yellow">RAG</span></td><td>${barRow(rag.avg_latency_ms||0, maxLat, 'yellow')}</td></tr>
          <tr><td><span class="blue">Sliding Win</span></td><td>${barRow(sw.avg_latency_ms||0, maxLat, 'blue')}</td></tr>
        </tbody>
      </table>
    </div>

    <div class="section">
      <div class="section-title">Avg Prefill Tokens</div>
      <table>
        <thead><tr><th>Method</th><th>Tokens</th></tr></thead>
        <tbody>
          <tr><td><span class="green">KV Memory</span></td><td>${barRow(kv.avg_prefill_tokens||0, maxTok, 'green')}</td></tr>
          <tr><td><span class="yellow">RAG</span></td><td>${barRow(rag.avg_prefill_tokens||0, maxTok, 'yellow')}</td></tr>
          <tr><td><span class="blue">Sliding Win</span></td><td>${barRow(sw.avg_prefill_tokens||0, maxTok, 'blue')}</td></tr>
        </tbody>
      </table>
    </div>

    ${kvBreakdown}

    ${answersSection}

    <div class="section">
      <div class="section-title">Run Config</div>
      <div class="config-grid">${cfgItems}</div>
    </div>
  `;
}

loadRuns();
setInterval(loadRuns, 10000);
</script>
</body>
</html>
"""


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>KV Memory Flow</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#191919;--muted:#667085;--line:#d8dde6;--soft:#eef1f5;--green:#0f766e;--red:#be123c;--amber:#a16207;--blue:#2563eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,Segoe UI,system-ui,sans-serif;letter-spacing:0}button,select{font:inherit;letter-spacing:0}
header{height:58px;display:flex;gap:12px;align-items:center;padding:12px 18px;background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:2}
.mark{width:18px;height:18px;border-radius:4px;border:1px solid var(--line);background:linear-gradient(135deg,var(--green),var(--red))}h1{font-size:18px;margin:0}.note{color:var(--muted);overflow-wrap:anywhere}.refresh{margin-left:auto;border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:7px 12px;cursor:pointer}
.layout{display:grid;grid-template-columns:minmax(260px,310px) 1fr;min-height:calc(100vh - 58px)}aside{padding:14px;background:#fbfcfd;border-right:1px solid var(--line);overflow:auto}main{padding:18px;overflow:auto}.eyebrow{font-size:12px;color:var(--muted);margin:0 0 8px}
.run{width:100%;display:block;text-align:left;border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:10px;margin:0 0 8px;cursor:pointer}.run:hover,.run.active{border-color:var(--green)}.rid{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}.rmeta{color:var(--muted);font-size:12px;margin-top:4px;overflow-wrap:anywhere}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--amber);margin-right:7px}.dot.completed{background:var(--green)}.dot.failed{background:var(--red)}
.empty,.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px}.empty{min-height:240px;display:grid;place-items:center;text-align:center;color:var(--muted);padding:20px}.panel{padding:14px;margin-bottom:14px}h2{font-size:16px;margin:0 0 10px}.sub{color:var(--muted);font-size:12px}.head{display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin-bottom:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:8px}.metric,.step,.box,.row{border:1px solid var(--line);border-radius:8px;background:#fcfcfd;padding:10px}.label{color:var(--muted);font-size:12px;margin-bottom:6px}.val{font-size:23px;font-weight:800;overflow-wrap:anywhere}.help{color:var(--muted);font-size:12px;margin-top:3px}.green{color:var(--green)}.red{color:var(--red)}.amber{color:var(--amber)}.blue{color:var(--blue)}
.flow{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:8px;margin-top:10px}.step{min-height:82px}.split{display:grid;grid-template-columns:minmax(230px,.75fr) minmax(320px,1.25fr);gap:12px}.list{display:grid;gap:8px}.row{overflow-wrap:anywhere}.bid{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--green);font-size:12px}.chip{display:inline-flex;min-height:22px;align-items:center;border:1px solid var(--line);background:var(--soft);border-radius:6px;padding:2px 7px;margin:2px 4px 2px 0;font-size:12px}.pick{background:#fce7f3;border-color:#f9a8d4;color:var(--red)}.hit{background:#ccfbf1;border-color:#5eead4;color:var(--green)}.warn{background:#fef3c7;border-color:#fcd34d;color:var(--amber)}
.qbar{display:grid;grid-template-columns:minmax(220px,1fr) auto auto;gap:8px;align-items:center}select{width:100%;min-height:38px;border:1px solid var(--line);border-radius:6px;background:var(--panel);padding:7px 9px}.answers{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:8px;overflow-wrap:anywhere}th{font-size:12px;color:var(--muted)}tr:last-child td{border-bottom:0}
@media(max-width:900px){.layout{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line);max-height:280px}.flow{grid-template-columns:repeat(2,1fr)}.split,.answers,.qbar{grid-template-columns:1fr}main{padding:12px}}
</style>
</head>
<body>
<header><img class="mark" alt="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="/><h1>KV Memory Flow</h1><span id="note" class="note">Pick a run</span><button class="refresh" onclick="loadRuns()">Refresh</button></header>
<div class="layout"><aside><p class="eyebrow">Runs</p><div id="runs">Loading...</div></aside><main id="main"><div class="empty">Pick a run to trace write, retrieval, rerank, and injection.</div></main></div>
<script>
const st={runs:[],id:null,trace:null,qid:null};
const x=v=>String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'","&#39;");
const f=(v,d=0)=>v===null||v===undefined||v===''?'-':(Number.isFinite(Number(v))?Number(v).toFixed(d):x(v));
const pct=v=>v===null||v===undefined||v===''?'-':f(Number(v)*100,0)+'%';
const ms=v=>v===null||v===undefined||v===''?'-':f(v,0)+' ms';
const sid=v=>{v=String(v||'');return v.length>12?v.slice(0,8)+'...'+v.slice(-4):(v||'-')};
const model=r=>String(r?.config?.model||r?.config?.model_id||'unknown').split(/[\\/]/).pop();
const chip=(t,c='')=>`<span class="chip ${c}">${x(t)}</span>`;
const metric=(l,v,h='',c='')=>`<div class="metric"><div class="label">${x(l)}</div><div class="val ${c}">${v}</div>${h?`<div class="help">${x(h)}</div>`:''}</div>`;
const step=(l,v,h='',c='')=>`<div class="step"><div class="label">${x(l)}</div><div class="val ${c}">${v}</div><div class="help">${x(h)}</div></div>`;

async function loadRuns(){
  const el=document.getElementById('runs'); el.innerHTML='Loading...';
  const data=await (await fetch('/api/runs')).json(); st.runs=data.runs||[];
  el.innerHTML=st.runs.length?st.runs.map(r=>`<button class="run ${r.run_id===st.id?'active':''}" onclick="selectRun('${x(r.run_id)}')"><div><span class="dot ${x(r.status)}"></span><span class="rid">${x(sid(r.run_id))}</span></div><div class="rmeta">${x(model(r))}</div><div class="rmeta">${x(r.started_at?new Date(r.started_at*1000).toLocaleString():'')}${r.summary?.kv_metrics?.overall_f1==null?'':' | F1 '+pct(r.summary.kv_metrics.overall_f1)}</div></button>`).join(''):'<div class="empty">No runs yet.</div>';
}
async function selectRun(id){
  st.id=id;st.qid=null;document.getElementById('main').innerHTML='<div class="empty">Loading trace...</div>';await loadRuns();
  const [rr,er]=await Promise.all([fetch(`/api/runs/${encodeURIComponent(id)}`),fetch(`/api/runs/${encodeURIComponent(id)}/events`)]);
  st.trace=buildTrace(await rr.json(),(await er.json()).events||[]);st.qid=st.trace.questions[0]?.question_id||null;render();
}
function buildTrace(run,events){
  const made=new Map(),stored=[],skips={},layers={},qs=new Map(),order=[];
  const key=e=>(e.question_id||'')+':'+(e.chunk_index||'');
  const addq=id=>{id=String(id||`q_${order.length+1}`);if(!qs.has(id)){qs.set(id,{question_id:id,question:'',question_type:'',gold_answer:'',predicted_answer:'',f1_score:null,em_score:null,correct:null,candidate_count:0,selected_count:0,selected_block_ids:[],retrieved_chunks:[],retrieval_diagnostics:{},stage1:{},stage2:{},fetch:{},generation:{}});order.push(id)}return qs.get(id)};
  for(const e of events){
    if(e.type==='write_chunk_created') made.set(key(e),e);
    if(e.type==='write_chunk_skipped') skips[e.reason||'unknown']=(skips[e.reason||'unknown']||0)+1;
    if(e.type==='write_chunk_stored'){const m=made.get(key(e))||{};(e.retrieval_layers||[]).forEach(l=>layers[l]=(layers[l]||0)+1);stored.push({block_id:e.block_id,question_id:e.question_id,chunk_index:e.chunk_index,total_chunks:e.total_chunks,token_count:e.token_count,importance_score:e.importance_score,retrieval_layers:e.retrieval_layers||[],chunk_preview:m.chunk_preview,seq:e.seq})}
  }
  for(const [i,r] of (run.summary?.kv_results||[]).entries()){const d=r.retrieval_diagnostics||{},q=addq(r.question_id||`summary_${i+1}`),sel=r.selected_block_ids||d.selected_ids||[];Object.assign(q,{question:r.question||q.question,question_type:r.question_type||q.question_type,gold_answer:r.gold_answer||'',predicted_answer:r.predicted_answer||'',f1_score:r.f1_score,em_score:r.em_score,correct:r.correct,candidate_count:r.candidate_count||d.candidate_count||0,selected_count:sel.length||d.selected_count||0,selected_block_ids:sel,retrieved_chunks:r.retrieved_chunks||[],retrieval_diagnostics:d,latency_ms:r.latency_ms,prefill_tokens:r.prefill_tokens});q.stage1.duration_ms=r.stage1_ms;q.stage2.duration_ms=r.stage2_ms;q.fetch.duration_ms=r.fetch_ms;q.generation.duration_ms=r.generate_ms}
  const types=new Set(['question_started','retrieval_stage1_done','retrieval_stage2_done','kv_fetch_done','generation_done','score_done','retrieval_diagnostics_done','question_finished']);
  for(const e of events){if(!types.has(e.type)||e.question_id==null)continue;const q=addq(e.question_id);if(e.question_type&&!q.question_type)q.question_type=e.question_type;
    if(e.type==='question_started')q.question=e.question||q.question;
    if(e.type==='retrieval_stage1_done'){q.stage1={duration_ms:e.duration_ms,candidate_count:e.candidate_count,query_token_count:e.query_token_count||e.retrieval_query_token_count};q.candidate_count=e.candidate_count||q.candidate_count}
    if(e.type==='retrieval_stage2_done'){const sel=e.selected_ids||q.selected_block_ids||[];q.stage2={duration_ms:e.duration_ms,selected_count:e.selected_count,selected_ids:sel};q.selected_block_ids=sel;q.selected_count=e.selected_count||sel.length}
    if(e.type==='kv_fetch_done')q.fetch={duration_ms:e.duration_ms,block_count:e.block_count,block_ids:e.block_ids,token_count:e.token_count};
    if(e.type==='generation_done')q.generation={duration_ms:e.duration_ms,output_chars:e.output_chars};
    if(e.type==='score_done'){q.em_score=e.em_score;q.f1_score=e.f1_score;q.correct=e.correct;q.predicted_answer=e.predicted_answer||q.predicted_answer;q.gold_answer=e.gold_answer||q.gold_answer}
    if(e.type==='retrieval_diagnostics_done')Object.assign(q.retrieval_diagnostics,{candidate_count:e.candidate_count,selected_count:e.selected_count,gold_in_stage1:e.gold_in_stage1,gold_in_selected:e.gold_in_selected,best_gold_rerank_rank:e.best_gold_rerank_rank,best_gold_stage1_rank:e.best_gold_stage1_rank});
    if(e.type==='question_finished'){q.latency_ms=e.latency_ms||q.latency_ms;q.prefill_tokens=e.prefill_tokens||q.prefill_tokens}
  }
  const ids=new Set(stored.map(b=>b.block_id).filter(Boolean));
  return {run,events:{count:events.length},ingest:{before:0,after:ids.size,created:made.size,stored:stored.length,skipped:Object.values(skips).reduce((a,b)=>a+b,0),skips,layers,tokens:stored.reduce((a,b)=>a+Number(b.token_count||0),0),blocks:stored.slice(0,200)},questions:order.map(id=>{const q=qs.get(id),d=q.retrieval_diagnostics||{};q.candidate_count=q.candidate_count||d.candidate_count||0;q.selected_count=q.selected_count||d.selected_count||(q.selected_block_ids||[]).length;q.top_candidates=d.top_candidates||[];q.gold_in_stage1=d.gold_in_stage1;q.gold_in_selected=d.gold_in_selected;return q})};
}
const qsel=()=>st.trace.questions.find(q=>q.question_id===st.qid)||st.trace.questions[0]||null;
function render(){const t=st.trace,r=t.run,ing=t.ingest,q=qsel();document.getElementById('note').textContent=`${r.status||'run'} | ${model(r)} | ${t.events.count} events`;document.getElementById('main').innerHTML=overview(t,q)+ingestion(ing)+retrieval(t,q)}
function overview(t,q){const m=t.run.summary?.kv_metrics||{},ing=t.ingest,inj=q?.fetch?.block_count??q?.selected_count??0,rerank=String(t.run.config?.stage2_reranker||'mmr').toUpperCase();return `<section class="panel"><div class="head"><div><h2>Run ${x(sid(t.run.run_id))}</h2><div class="sub">${x(t.run.metadata?.session_id||'')}</div></div><div class="sub">${x(t.run.status||'')}</div></div><div class="grid">${metric('KV F1',pct(m.overall_f1),'answer quality','green')}${metric('Avg latency',ms(m.avg_latency_ms),'read path')}${metric('Questions',f(t.run.summary?.question_count??t.questions.length),'scored')}${metric('Stored blocks',f(ing.after),'run-local')}</div><div class="flow">${step('Before ingestion',f(ing.before),'run-local blocks')}${step('After write',f(ing.after),f(ing.tokens)+' tokens','green')}${step('Stage 1 candidates',f(q?.candidate_count),'from stored blocks','blue')}${step(rerank+' pick',f(q?.selected_count),'selected for injection','amber')}${step('Injected',f(inj),f(q?.fetch?.token_count)+' KV tokens','red')}</div></section>`}
function ingestion(ing){const reasons=Object.entries(ing.skips||{}),layers=Object.entries(ing.layers||{}).map(([l,c])=>chip('L'+l+': '+c)).join('')||'<span class="sub">No stored layers yet</span>',blocks=(ing.blocks||[]).slice(0,10).map(b=>`<div class="row"><div><span class="bid">${x(sid(b.block_id))}</span> ${chip('q '+(b.question_id??'-'))} ${chip(f(b.token_count)+' tokens')}</div><div class="sub">${x(b.chunk_preview||'No chunk preview captured')}</div><div>${(b.retrieval_layers||[]).map(l=>chip('L'+l)).join('')}</div></div>`).join('')||'<div class="sub">No KV blocks written in this run.</div>';return `<section class="panel"><div class="head"><div><h2>Ingestion</h2><div class="sub">chunk -> capture -> KV store -> vector index</div></div><div class="sub">${f(ing.created)} chunks seen</div></div><div class="split"><div><div class="grid">${metric('Stored',f(ing.stored),'KV blocks','green')}${metric('Skipped',f(ing.skipped),'gates and duplicates',ing.skipped?'amber':'')}</div><h2 style="margin-top:12px">Skipped reasons</h2><div class="list">${reasons.length?reasons.map(([r,c])=>`<div class="row"><span>${x(r)}</span><strong style="float:right">${f(c)}</strong></div>`).join(''):'<div class="sub">No skipped chunks.</div>'}</div><h2 style="margin-top:12px">Layers</h2><div>${layers}</div></div><div><h2>Stored blocks</h2><div class="list">${blocks}</div></div></div></section>`}
function retrieval(t,q){if(!q)return '<section class="panel"><h2>Retrieval</h2><div class="sub">No question traces in this run.</div></section>';const opts=t.questions.map(it=>`<option value="${x(it.question_id)}" ${it.question_id===q.question_id?'selected':''}>${x(it.question_id+' | '+(it.question||it.question_type||'question'))}</option>`).join('');return `<section class="panel"><h2>Retrieval</h2><div class="qbar"><select onchange="st.qid=this.value;render()">${opts}</select>${chip('F1 '+pct(q.f1_score),q.correct?'hit':'warn')}${chip(f(q.candidate_count)+' of '+f(t.ingest.after)+' candidates')}</div><p><strong>Query</strong><br><span class="sub">${x(q.question||'No question text captured')}</span></p><div class="answers"><div class="box"><strong>Gold</strong><br>${x(q.gold_answer||'-')}</div><div class="box"><strong>Predicted</strong><br>${x(q.predicted_answer||'-')}</div></div><div class="grid">${metric('Stage 1',ms(q.stage1?.duration_ms),f(q.stage1?.query_token_count)+' query tokens')}${metric('Rerank',ms(q.stage2?.duration_ms),f(q.selected_count)+' selected')}${metric('Fetch',ms(q.fetch?.duration_ms),f(q.fetch?.block_count)+' blocks')}${metric('Inject + generate',ms(q.generation?.duration_ms),f(q.generation?.output_chars)+' chars')}</div><div style="height:12px"></div>${candidates(q)}</section>`}
function candidates(q){const rows=q.top_candidates||[],sel=new Set(q.selected_block_ids||q.retrieval_diagnostics?.selected_ids||[]);if(!rows.length)return '<div class="empty">No candidate diagnostics captured for this question.</div>';return `<div style="overflow-x:auto"><table><thead><tr><th>Stage 1</th><th>Rerank</th><th>Score</th><th>Answer</th><th>Block</th><th>Chunk</th></tr></thead><tbody>${rows.map(r=>{const picked=r.selected||sel.has(r.block_id),hit=r.gold_substring||Number(r.gold_overlap||0)>0;return `<tr><td>#${f(r.stage1_rank)}</td><td>#${f(r.rerank_rank)}</td><td>${f(r.relevance,3)}</td><td>${hit?chip('match','hit'):chip('no match')}</td><td><div class="bid">${x(sid(r.block_id))}</div>${picked?chip('picked','pick'):chip('not picked')} ${chip(f(r.token_count)+' tokens')} ${r.importance_score==null?'':chip('importance '+f(r.importance_score,2))}</td><td>${x(r.chunk_preview||'')}<div class="sub">${(r.gold_overlap_terms||[]).map(x).join(', ')}</div></td></tr>`}).join('')}</tbody></table></div>`}
loadRuns();
</script>
</body>
</html>
"""


def create_app(
    *,
    obs_dir: str = ".kvmem_obs",
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse

    store = ObservabilityStore(obs_dir)
    vector_db = VectorDB(url=qdrant_url, port=qdrant_port)
    app = FastAPI(title="KV Memory Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return HTMLResponse(_DASHBOARD_HTML)

    @app.get("/api/runs")
    def list_runs(
        status: Optional[str] = None,
        model: Optional[str] = None,
        dtype: Optional[str] = None,
        dataset: Optional[str] = None,
        synthetic: Optional[bool] = None,
        n: Optional[int] = None,
        start_after: Optional[str] = None,
        start_before: Optional[str] = None,
    ):
        runs = store.list_runs(
            status=status,
            model=model,
            dtype=dtype,
            dataset=dataset,
            synthetic=synthetic,
            n=n,
            start_after=_parse_time(start_after),
            start_before=_parse_time(start_before),
        )
        return {"runs": runs}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.post("/api/runs/{run_id}/status")
    def update_run_status(run_id: str, body: dict):
        status = body.get("status")
        if status not in ("failed", "completed"):
            raise HTTPException(
                status_code=422,
                detail="status must be 'failed' or 'completed'",
            )
        try:
            run = store.get_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found")
        error = body.get("error")
        store.finish_run(run_id, status=status, error=error)
        return {"run_id": run_id, "status": status}

    @app.get("/api/runs/{run_id}/events")
    def get_events(
        run_id: str,
        event_type: Optional[str] = Query(default=None, alias="type"),
        phase: Optional[str] = None,
        question_id: Optional[str] = None,
        level: Optional[str] = None,
        since_ts: Optional[str] = None,
        since_seq: Optional[int] = None,
        search: Optional[str] = None,
        limit: Optional[int] = 1000,
    ):
        if not store.get_run(run_id):
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "events": store.get_events(
                run_id,
                event_type=event_type,
                phase=phase,
                question_id=question_id,
                level=level,
                since_ts=_parse_time(since_ts),
                since_seq=since_seq,
                search=search,
                limit=limit,
            )
        }

    @app.get("/api/live/{run_id}")
    def get_live(
        run_id: str,
        since_ts: Optional[str] = None,
        since_seq: Optional[int] = None,
        limit: Optional[int] = 500,
    ):
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run": run,
            "events": store.get_live_events(
                run_id,
                since_ts=_parse_time(since_ts),
                since_seq=since_seq,
                limit=limit,
            ),
        }

    @app.get("/api/qdrant/collections")
    def list_collections():
        return {"collections": vector_db.list_collections()}

    @app.get("/api/qdrant/points")
    def list_points(
        collection: str,
        limit: int = 100,
        offset: Optional[str] = None,
        with_vectors: bool = False,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        question_id: Optional[str] = None,
        phase: Optional[str] = None,
        agent_id: Optional[str] = None,
        shared: Optional[bool] = None,
        importance_min: Optional[float] = None,
        importance_max: Optional[float] = None,
        token_count_min: Optional[int] = None,
        token_count_max: Optional[int] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        text_contains: Optional[str] = None,
        layer: Optional[int] = None,
    ):
        return vector_db.scroll_points(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_vectors=with_vectors,
            run_id=run_id,
            session_id=session_id,
            question_id=question_id,
            phase=phase,
            agent_id=agent_id,
            shared=shared,
            importance_min=importance_min,
            importance_max=importance_max,
            token_count_min=token_count_min,
            token_count_max=token_count_max,
            created_after=_parse_time(created_after),
            created_before=_parse_time(created_before),
            text_contains=text_contains,
            layer=layer,
        )

    return app


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="KV Memory Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--obs-dir", default=".kvmem_obs")
    parser.add_argument("--qdrant-url", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    args = parser.parse_args()

    app = create_app(
        obs_dir=args.obs_dir,
        qdrant_url=args.qdrant_url,
        qdrant_port=args.qdrant_port,
    )
    print(f"Dashboard -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
