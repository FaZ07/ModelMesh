const BASE = '/api'

async function j(res) {
  if (!res.ok) {
    const d = await res.json().catch(() => ({}))
    throw new Error(d.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  health: () => fetch(`${BASE}/health`).then(j),
  listModels: () => fetch(`${BASE}/models`).then(j),

  register: (name, target, modelFile, csvFile) => {
    const fd = new FormData()
    fd.append('name', name)
    fd.append('target', target)
    fd.append('model', modelFile)
    fd.append('train_csv', csvFile)
    return fetch(`${BASE}/models`, { method: 'POST', body: fd }).then(j)
  },

  predict: (id, features) =>
    fetch(`${BASE}/models/${id}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    }).then(j),

  drift: (id) => fetch(`${BASE}/models/${id}/drift`).then(j),

  retrain: (id, target, csvFile) => {
    const fd = new FormData()
    fd.append('target', target)
    fd.append('labelled_csv', csvFile)
    return fetch(`${BASE}/models/${id}/retrain`, { method: 'POST', body: fd }).then(j)
  },

  job: (jobId) => fetch(`${BASE}/jobs/${jobId}`).then(j),
}
