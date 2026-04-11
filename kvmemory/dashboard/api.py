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
    print(f"Dashboard → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
