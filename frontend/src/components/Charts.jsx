/* Dependency-free inline-SVG charts — keeps the app fully offline. */

const SEV = {
  stable: '#34d399',
  moderate: '#fbbf24',
  significant: '#fb7185',
}

export function PsiBars({ features }) {
  if (!features?.length) return null
  const max = Math.max(0.4, ...features.map((f) => f.psi))
  return (
    <div className="space-y-2">
      {features.map((f) => (
        <div key={f.feature} className="flex items-center gap-3">
          <div className="w-40 truncate text-xs font-mono text-slate-400" title={f.feature}>
            {f.feature}
          </div>
          <div className="flex-1 h-6 bg-ink rounded-md overflow-hidden relative">
            <div
              className="h-full rounded-md transition-all"
              style={{ width: `${(f.psi / max) * 100}%`, background: SEV[f.status] }}
            />
            {/* PSI 0.1 / 0.25 threshold markers */}
            {[0.1, 0.25].map((t) => (
              <div key={t} className="absolute top-0 h-full border-l border-dashed border-slate-600"
                   style={{ left: `${(t / max) * 100}%` }} />
            ))}
          </div>
          <div className="w-16 text-right text-xs font-mono font-bold"
               style={{ color: SEV[f.status] }}>
            {f.psi.toFixed(3)}
          </div>
        </div>
      ))}
      <div className="flex gap-4 text-[10px] text-slate-500 pt-1 ml-[172px]">
        <span>┊ 0.10 moderate</span><span>┊ 0.25 significant</span>
      </div>
    </div>
  )
}

export function DistOverlay({ hist, feature }) {
  if (!hist) return null
  const W = 460, H = 150, pad = 24
  const n = hist.current.length
  const max = Math.max(...hist.reference, ...hist.current, 0.001)
  const bw = (W - pad * 2) / n

  const bars = (vals, color, offset, w) =>
    vals.map((vv, i) => {
      const h = (vv / max) * (H - pad * 2)
      return (
        <rect key={`${color}${i}`} x={pad + i * bw + offset} y={H - pad - h}
              width={w} height={Math.max(0, h)} fill={color} rx="1.5" />
      )
    })

  return (
    <div>
      <div className="k mb-1">{feature} · reference vs live</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#334155" />
        {bars(hist.reference, '#475569', 1, bw * 0.42)}
        {bars(hist.current, '#a78bfa', bw * 0.5, bw * 0.42)}
      </svg>
      <div className="flex gap-4 text-[11px] mt-1">
        <span className="flex items-center gap-1"><i className="w-3 h-3 inline-block rounded-sm bg-slate-600" /> reference (training)</span>
        <span className="flex items-center gap-1"><i className="w-3 h-3 inline-block rounded-sm bg-viol-400" /> current (live)</span>
      </div>
    </div>
  )
}

export function Gauge({ value, status }) {
  const pct = Math.min(1, value / 0.5)
  const r = 52, c = 2 * Math.PI * r
  return (
    <div className="relative w-36 h-36">
      <svg viewBox="0 0 130 130" className="-rotate-90 w-full h-full">
        <circle cx="65" cy="65" r={r} fill="none" stroke="#1e293b" strokeWidth="12" />
        <circle cx="65" cy="65" r={r} fill="none" stroke={SEV[status]} strokeWidth="12"
                strokeLinecap="round" strokeDasharray={c}
                strokeDashoffset={c * (1 - pct)} />
      </svg>
      <div className="absolute inset-0 grid place-items-center">
        <div className="text-center">
          <div className="text-2xl font-black font-mono text-white">{value.toFixed(3)}</div>
          <div className="text-[10px] uppercase font-bold tracking-wider"
               style={{ color: SEV[status] }}>
            {status}
          </div>
        </div>
      </div>
    </div>
  )
}
