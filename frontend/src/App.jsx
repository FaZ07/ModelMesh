import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import { DistOverlay, Gauge, PsiBars } from './components/Charts'

/* ───────────────────────── header ───────────────────────── */
function Header({ health }) {
  return (
    <header className="flex items-center justify-between flex-wrap gap-4 mb-8">
      <div>
        <h1 className="text-3xl font-black text-white tracking-tight">⚙️ ModelMesh</h1>
        <p className="text-slate-400 text-sm mt-1">
          Serve any scikit-learn model · watch it drift in real time · auto-retrain when it decays · 100% local
        </p>
      </div>
      {health && (
        <div className="card px-5 py-3 text-sm">
          <span className="k">Registered models</span>
          <div className="v">{health.models}</div>
        </div>
      )}
    </header>
  )
}

/* ───────────────────────── register ───────────────────────── */
function Register({ onDone }) {
  const [name, setName] = useState('')
  const [target, setTarget] = useState('target')
  const [model, setModel] = useState(null)
  const [csv, setCsv] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const submit = async () => {
    setBusy(true); setErr('')
    try {
      await api.register(name || 'untitled', target, model, csv)
      setName(''); setModel(null); setCsv(null); onDone()
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <section className="card p-6">
      <h2 className="text-lg font-bold text-white mb-1">Register a model</h2>
      <p className="text-slate-400 text-sm mb-4">
        Upload any pickled sklearn estimator + its training CSV. ModelMesh derives the input
        schema and per-feature drift reference automatically.
      </p>
      <div className="grid sm:grid-cols-2 gap-3">
        <input className="bg-ink border border-edge rounded-lg px-3 py-2 text-sm"
               placeholder="Model name" value={name} onChange={(e) => setName(e.target.value)} />
        <input className="bg-ink border border-edge rounded-lg px-3 py-2 text-sm font-mono"
               placeholder="target column (e.g. target)" value={target}
               onChange={(e) => setTarget(e.target.value)} />
        <label className="btn-ghost cursor-pointer text-sm">
          {model ? model.name : 'model.joblib'}
          <input type="file" className="hidden" accept=".joblib,.pkl,.pickle"
                 onChange={(e) => setModel(e.target.files[0])} />
        </label>
        <label className="btn-ghost cursor-pointer text-sm">
          {csv ? csv.name : 'train.csv'}
          <input type="file" className="hidden" accept=".csv"
                 onChange={(e) => setCsv(e.target.files[0])} />
        </label>
      </div>
      <div className="flex items-center gap-3 mt-4">
        <button className="btn-primary" disabled={busy || !model || !csv} onClick={submit}>
          {busy ? 'Registering…' : 'Register'}
        </button>
        {err && <span className="text-rose-400 text-sm">{err}</span>}
      </div>
    </section>
  )
}

/* ───────────────────────── predict ───────────────────────── */
function Predictor({ model }) {
  const [vals, setVals] = useState({})
  const [out, setOut] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => { setVals({}); setOut(null) }, [model.id])

  const run = async () => {
    setErr('')
    try {
      const feats = Object.fromEntries(model.features.map((f) => [f, Number(vals[f] ?? 0)]))
      setOut(await api.predict(model.id, feats))
    } catch (e) { setErr(e.message) }
  }

  return (
    <div className="card p-5">
      <h3 className="font-bold text-white mb-3">Live prediction</h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 max-h-56 overflow-y-auto pr-1">
        {model.features.map((f) => (
          <div key={f}>
            <div className="text-[10px] text-slate-500 font-mono truncate" title={f}>{f}</div>
            <input type="number" step="any"
              className="w-full bg-ink border border-edge rounded px-2 py-1 text-xs font-mono"
              value={vals[f] ?? ''} onChange={(e) => setVals({ ...vals, [f]: e.target.value })} />
          </div>
        ))}
      </div>
      <div className="flex items-center gap-4 mt-4">
        <button className="btn-primary" onClick={run}>Predict</button>
        {out && (
          <div className="flex gap-6">
            <div><span className="k">prediction</span><div className="v text-viol-400">{out.prediction}</div></div>
            {out.confidence != null && (
              <div><span className="k">confidence</span><div className="v">{(out.confidence * 100).toFixed(1)}%</div></div>
            )}
          </div>
        )}
        {err && <span className="text-rose-400 text-sm">{err}</span>}
      </div>
    </div>
  )
}

/* ───────────────────────── retrain ───────────────────────── */
function Retrain({ model, onPromoted }) {
  const [target, setTarget] = useState('target')
  const [csv, setCsv] = useState(null)
  const [job, setJob] = useState(null)
  const timer = useRef(null)

  const start = async () => {
    const { job_id } = await api.retrain(model.id, target, csv)
    timer.current = setInterval(async () => {
      const j = await api.job(job_id)
      setJob(j)
      if (j.status === 'done' || j.status === 'failed') {
        clearInterval(timer.current)
        if (j.result?.promoted) onPromoted()
      }
    }, 800)
  }
  useEffect(() => () => clearInterval(timer.current), [])

  return (
    <div className="card p-5">
      <h3 className="font-bold text-white mb-1">Auto-retrain (shadow → promote)</h3>
      <p className="text-slate-400 text-xs mb-3">
        Fits a challenger on fresh labelled data; promotes only if it beats the champion on holdout.
      </p>
      <div className="flex flex-wrap gap-2 items-center">
        <input className="bg-ink border border-edge rounded px-2 py-1 text-xs font-mono w-32"
               value={target} onChange={(e) => setTarget(e.target.value)} />
        <label className="btn-ghost cursor-pointer text-xs">
          {csv ? csv.name : 'fresh_labelled.csv'}
          <input type="file" className="hidden" accept=".csv"
                 onChange={(e) => setCsv(e.target.files[0])} />
        </label>
        <button className="btn-primary text-sm" disabled={!csv} onClick={start}>Trigger retrain</button>
      </div>
      {job && (
        <div className="mt-4 bg-ink rounded-lg p-3 font-mono text-[11px] space-y-1 max-h-44 overflow-y-auto">
          {job.log.map((l, i) => (
            <div key={i} className="text-slate-400">
              <span className="text-slate-600">[{l.t}s]</span> {l.msg}
            </div>
          ))}
          {job.result && (
            <div className={`mt-2 font-bold ${job.result.promoted ? 'text-emerald-400' : 'text-amber-400'}`}>
              {job.result.promoted
                ? `✅ Promoted v${job.result.new_version} · ${job.result.champion_score} → ${job.result.challenger_score}`
                : job.result.error
                  ? `✕ ${job.result.error}`
                  : `✋ Kept champion · ${job.result.champion_score} ≥ ${job.result.challenger_score}`}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ───────────────────────── drift dashboard ───────────────────────── */
function DriftPanel({ model }) {
  const [d, setD] = useState(null)
  const [auto, setAuto] = useState(true)
  const timer = useRef(null)

  const pull = () => api.drift(model.id).then(setD).catch(() => {})
  useEffect(() => {
    pull()
    if (auto) timer.current = setInterval(pull, 4000)
    return () => clearInterval(timer.current)
  }, [model.id, auto])

  if (!d) return <div className="card p-6 text-slate-500 text-sm">Loading drift…</div>

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold text-white">Drift monitor</h3>
        <label className="text-xs text-slate-400 flex items-center gap-2">
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
          auto-refresh 4s
        </label>
      </div>

      {!d.ready ? (
        <p className="text-slate-400 text-sm">
          {d.message} <span className="text-slate-600">({d.total_predictions} total logged)</span>
        </p>
      ) : (
        <>
          <div className="flex flex-wrap gap-8 items-center mb-6">
            <Gauge value={d.overall.max_psi} status={d.overall.status} />
            <div className="grid grid-cols-2 gap-x-8 gap-y-3">
              <div><span className="k">samples</span><div className="v">{d.samples}</div></div>
              <div><span className="k">total logged</span><div className="v">{d.total_predictions}</div></div>
              <div><span className="k">drifting features</span><div className="v">{d.overall.drifting_features}</div></div>
              <div>
                <span className="k">stream (ADWIN)</span>
                <div className={`v ${d.stream.adwin_drift ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {d.stream.adwin_drift ? 'CHANGE' : 'steady'}
                </div>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            <div>
              <div className="k mb-3">Population Stability Index — per feature</div>
              <PsiBars features={d.features} />
            </div>
            <DistOverlay hist={d.features[0]?.ref_hist} feature={d.features[0]?.feature} />
          </div>
        </>
      )}
    </div>
  )
}

/* ───────────────────────── app ───────────────────────── */
export default function App() {
  const [health, setHealth] = useState(null)
  const [models, setModels] = useState([])
  const [sel, setSel] = useState(null)

  const refresh = async () => {
    const h = await api.health().catch(() => null)
    setHealth(h)
    const m = await api.listModels().catch(() => [])
    setModels(m)
    setSel((cur) => m.find((x) => x.id === cur?.id) || m[0] || null)
  }
  useEffect(() => { refresh() }, [])

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <Header health={health} />

      <div className="grid lg:grid-cols-[260px_1fr] gap-6">
        {/* model list */}
        <aside className="space-y-2">
          <div className="k mb-2">Models</div>
          {models.length === 0 && (
            <p className="text-slate-600 text-xs">None yet — register one →</p>
          )}
          {models.map((m) => (
            <button key={m.id} onClick={() => setSel(m)}
              className={`w-full text-left p-3 rounded-xl border transition ${
                sel?.id === m.id
                  ? 'bg-viol-500/15 border-viol-500/50'
                  : 'bg-panel border-edge hover:border-slate-600'}`}>
              <div className="font-semibold text-white text-sm truncate">{m.name}</div>
              <div className="text-[11px] text-slate-500 font-mono">
                {m.kind} · v{m.version} · {m.task.slice(0, 4)}
              </div>
            </button>
          ))}
        </aside>

        <main className="space-y-6">
          <Register onDone={refresh} />
          {sel && (
            <>
              <div className="card p-5 flex flex-wrap gap-6 items-center">
                <div>
                  <span className="k">active model</span>
                  <div className="text-xl font-black text-white">{sel.name}</div>
                </div>
                <span className="pill bg-viol-500/20 text-viol-300">{sel.kind}</span>
                <span className="pill bg-edge text-slate-300">v{sel.version}</span>
                <span className="pill bg-edge text-slate-300">{sel.task}</span>
                <span className="pill bg-edge text-slate-300">{sel.features.length} features</span>
                {sel.metrics && (
                  <span className="pill bg-emerald-500/15 text-emerald-300">
                    holdout {sel.metrics.holdout_score}
                  </span>
                )}
              </div>
              <Predictor model={sel} />
              <DriftPanel model={sel} key={sel.id + sel.version} />
              <Retrain model={sel} onPromoted={refresh} />
            </>
          )}
        </main>
      </div>

      <footer className="text-center text-slate-600 text-xs mt-12">
        ModelMesh · PSI + Jensen–Shannon + ADWIN drift detection · shadow-promote retraining · fully offline · Mohamed Fazil
      </footer>
    </div>
  )
}
