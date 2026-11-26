"use client";
import { useCallback, useEffect, useState } from "react";

function readToken() {
  try { return localStorage.getItem("awaaz_admin_token") || ""; } catch { return ""; }
}
function saveToken(t) {
  try { localStorage.setItem("awaaz_admin_token", t); } catch { /* private mode */ }
}
const fmt = (v, d = 3) => (v === null || v === undefined ? "-" : Number(v).toFixed(d));
const ms = (v) => (v === undefined ? "-" : `${Math.round(v)} ms`);

function ModelCard({ title, info, stats }) {
  const loaded = info && info.loaded;
  return (
    <div className="panel">
      <h2>{title} <span className={`pill ${loaded ? "ok" : "idle"}`}>{loaded ? `v${info.version}` : "no model"}</span></h2>
      <dl className="kv">
        <dt>Registry stage</dt><dd>{info?.stage || "-"}</dd>
        <dt>Requests</dt><dd>{stats?.requests ?? 0}</dd>
        <dt>Latency p95</dt><dd>{ms(stats?.latency_p95_ms)}</dd>
        <dt>Flagged as fake</dt><dd>{stats?.fake_rate !== undefined ? `${Math.round(stats.fake_rate * 100)}%` : "-"}</dd>
        <dt>EER on feedback</dt><dd>{fmt(stats?.eer_on_feedback)} <span className="muted">({stats?.labelled ?? 0} labelled)</span></dd>
      </dl>
    </div>
  );
}

export default function Dashboard() {
  const [token, setToken] = useState("");
  const [draft, setDraft] = useState("");
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [preds, setPreds] = useState([]);
  const [weight, setWeight] = useState(0);
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const [links, setLinks] = useState({ grafana: "", mlflow: "" });

  useEffect(() => { const t = readToken(); setToken(t); setDraft(t); }, []);
  useEffect(() => {
    fetch("/api/config").then((r) => r.json()).then((c) => setLinks({
      grafana: c.grafana_url || `${window.location.protocol}//${window.location.hostname}:3001`,
      mlflow: c.mlflow_url || "",
    })).catch(() => {});
  }, []);

  const load = useCallback(async () => {
    if (!token) return;
    const h = { "X-Admin-Token": token };
    try {
      const [s, hl, p] = await Promise.all([
        fetch("/api/admin/stats", { headers: h }),
        fetch("/api/health"),
        fetch("/api/admin/predictions?limit=25", { headers: h }),
      ]);
      if (s.status === 401) { setErr("The admin token is not valid."); return; }
      if (!s.ok) throw new Error(`HTTP ${s.status}`);
      const sj = await s.json();
      setStats(sj); setHealth(await hl.json()); setPreds(await p.json());
      setWeight((w) => (w === 0 ? sj.canary_weight : w));
      setErr("");
    } catch {
      setErr("The gateway is not reachable.");
    }
  }, [token]);

  useEffect(() => { load(); const id = setInterval(load, 10000); return () => clearInterval(id); }, [load]);

  async function post(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "X-Admin-Token": token, ...(body ? { "Content-Type": "application/json" } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
    return r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`));
  }
  async function applyWeight(w, action) {
    await post("/api/admin/canary", { weight: w, action, detail: "set from dashboard" });
    setWeight(w); setMsg(`Candidate now receives ${Math.round(w * 100)}% of traffic.`); load();
  }
  async function reload(variant) {
    const r = await post(`/api/admin/reload/${variant}`);
    setMsg(r.loaded ? `${variant} reloaded: v${r.version}` : `${variant}: no model in its registry stage`); load();
  }

  if (!token) {
    return (
      <div className="panel" style={{ maxWidth: 520 }}>
        <h2>Sign in to operations</h2>
        <p className="muted" style={{ marginBottom: 14 }}>Enter the ADMIN_TOKEN configured on the gateway.</p>
        <form className="token" onSubmit={(e) => { e.preventDefault(); saveToken(draft); setToken(draft); }}>
          <input type="password" value={draft} onChange={(e) => setDraft(e.target.value)} aria-label="Admin token" />
          <button className="btn primary" type="submit">Open dashboard</button>
        </form>
      </div>
    );
  }

  const d = stats?.drift;
  return (
    <>
      <div className="ops-head">
        <div>
          <h1>Operations</h1>
          <p>{stats ? `${stats.total_predictions} checks served, ${stats.total_feedback} labelled by users` : "Loading..."}</p>
        </div>
        <div className="links">
          {links.grafana && <a href={links.grafana} target="_blank" rel="noreferrer">Grafana dashboards</a>}
          {links.mlflow && <a href={links.mlflow} target="_blank" rel="noreferrer">MLflow experiments</a>}
          <a href={`/api/admin/drift-report?token=${encodeURIComponent(token)}`} target="_blank" rel="noreferrer">Latest drift report</a>
          <button className="btn" onClick={() => { saveToken(""); setToken(""); }}>Sign out</button>
        </div>
      </div>
      {err && <p className="error" style={{ marginBottom: 16 }}>{err}</p>}
      {msg && <p className="muted" role="status" style={{ marginBottom: 16 }}>{msg}</p>}

      <div className="grid2">
        <ModelCard title="Champion" info={health?.champion} stats={stats?.champion} />
        <ModelCard title="Candidate" info={health?.candidate} stats={stats?.candidate} />
      </div>

      <div className="grid2">
        <div className="panel">
          <h2>Canary rollout</h2>
          <p className="muted">Share of traffic routed to the candidate model. The canary flow promotes or rolls back automatically.</p>
          <div className="slider">
            <input type="range" min="0" max="1" step="0.05" value={weight}
                   onChange={(e) => setWeight(Number(e.target.value))} aria-label="Candidate traffic share" />
            <strong>{Math.round(weight * 100)}%</strong>
          </div>
          <div className="row">
            <button className="btn primary" onClick={() => applyWeight(weight, "weight")}>Apply traffic split</button>
            <button className="btn" onClick={() => applyWeight(0, "stop")}>Stop canary</button>
            <button className="btn" onClick={() => reload("champion")}>Reload champion</button>
            <button className="btn" onClick={() => reload("candidate")}>Reload candidate</button>
          </div>
        </div>
        <div className="panel">
          <h2>Drift <span className={`pill ${!d ? "idle" : d.drift_detected ? "bad" : "ok"}`}>{!d ? "no check yet" : d.drift_detected ? "drift detected" : "stable"}</span></h2>
          {d ? (
            <dl className="kv">
              <dt>Checked</dt><dd>{new Date(d.created_at).toLocaleString()}</dd>
              <dt>Clips compared</dt><dd>{d.n_current}</dd>
              <dt>MMD p-value</dt><dd>{fmt(d.mmd_p_value, 4)}</dd>
              <dt>Out-of-distribution</dt><dd>{Math.round(d.ood_rate * 100)}% of clips</dd>
              <dt>Drifted features</dt><dd>{Math.round(d.drifted_features_share * 100)}%</dd>
            </dl>
          ) : <p className="muted">The monitor runs every 15 minutes once at least 50 checks have been served.</p>}
        </div>
      </div>

      <div className="panel" style={{ marginBottom: 20 }}>
        <h2>Recent checks</h2>
        <div className="tablewrap">
          <table>
            <thead><tr><th>#</th><th>Time</th><th>Verdict</th><th>P(fake)</th><th>Model</th><th>Length</th><th>Latency</th><th>User label</th></tr></thead>
            <tbody>
              {preds.map((p) => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{new Date(p.created_at).toLocaleTimeString()}</td>
                  <td>{p.verdict.replace("_", " ")}</td>
                  <td>{fmt(p.prob_fake, 2)}</td>
                  <td>v{p.model_version} {p.variant === "candidate" ? "(canary)" : ""}</td>
                  <td>{p.duration_s.toFixed(1)} s</td>
                  <td>{ms(p.latency_ms)}</td>
                  <td>{p.feedback || "-"}</td>
                </tr>
              ))}
              {!preds.length && <tr><td colSpan="8" className="muted">No checks yet. Upload a recording on the Check a voice page.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <h2>Rollout history</h2>
        <div className="tablewrap">
          <table>
            <thead><tr><th>Time</th><th>Action</th><th>Candidate share</th><th>Detail</th></tr></thead>
            <tbody>
              {(stats?.canary_events || []).map((e, i) => (
                <tr key={i}><td>{new Date(e.at).toLocaleString()}</td><td>{e.action}</td><td>{Math.round(e.weight * 100)}%</td><td>{e.detail}</td></tr>
              ))}
              {!stats?.canary_events?.length && <tr><td colSpan="4" className="muted">No rollouts yet.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
