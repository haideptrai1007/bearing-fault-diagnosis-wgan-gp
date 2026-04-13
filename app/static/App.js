const MODEL_KEYS = ['MobileOneS0', 'MobileNetV4', 'GhostNetV2', 'EdgeNeXtXXS'];

const MODEL_PATHS = {
  cwt: {
    MobileOneS0: "D:\\Capstone\\models\\ONNX\\CNN\\SCALOGRAM\\MobileOneS0\\best_model.onnx",
    MobileNetV4: "D:\\Capstone\\models\\ONNX\\CNN\\SCALOGRAM\\MobileNetV4\\best_model.onnx",
    GhostNetV2:  "D:\\Capstone\\models\\ONNX\\CNN\\SCALOGRAM\\GhostNetV2\\best_model.onnx",
    EdgeNeXtXXS: "D:\\Capstone\\models\\ONNX\\CNN\\SCALOGRAM\\EdgenextXXS\\best_model.onnx",
  },
  stft: {
    MobileOneS0: "D:\\Capstone\\models\\ONNX\\CNN\\SPECTROGRAM\\MobileOneS0\\best_model.onnx",
    MobileNetV4: "D:\\Capstone\\models\\ONNX\\CNN\\SPECTROGRAM\\MobileNetV4\\best_model.onnx",
    GhostNetV2:  "D:\\Capstone\\models\\ONNX\\CNN\\SPECTROGRAM\\GhostNetV2\\best_model.onnx",
    EdgeNeXtXXS: "D:\\Capstone\\models\\ONNX\\CNN\\SPECTROGRAM\\EdgenextXXS\\best_model.onnx",
  },
};

const FAULT_MAP = {
  0: { name: 'Normal condition', code: 'Class 0 — No fault detected',      icon: '✓', cls: 'normal' },
  1: { name: 'Outer race fault', code: 'Class 1 — Outer race defect',      icon: '⚠', cls: 'outer'  },
  2: { name: 'Inner race fault', code: 'Class 2 — Inner race defect',      icon: '⚡', cls: 'inner'  },
  3: { name: 'Ball fault',       code: 'Class 3 — Rolling element defect', icon: '◉', cls: 'ball'   },
};

const TRANSFORM_LABELS = {
  cwt:  'Continuous Wavelet Transform · Scalogram',
  stft: 'Short-Time Fourier Transform · Spectrogram',
};

let currentTransform = 'cwt';
let currentModel     = 'MobileOneS0';
let isRunning        = false;

// ── Transform ────────────────────────────────────────────────
function switchTransform(t) {
  if (currentTransform === t) return;
  currentTransform = t;
  document.getElementById('tab-cwt').classList.toggle('active',  t === 'cwt');
  document.getElementById('tab-stft').classList.toggle('active', t === 'stft');
  document.getElementById('transform-sub').textContent = TRANSFORM_LABELS[t];
  document.getElementById('transform-plot-label').textContent =
    t === 'cwt' ? 'Scalogram (CWT)' : 'Spectrogram (STFT)';
}

// ── Model ────────────────────────────────────────────────────
function selectModel(m) {
  if (currentModel === m) return;
  currentModel = m;
  MODEL_KEYS.forEach(name =>
    document.getElementById('btn-' + name).classList.toggle('active', name === m)
  );
}

// ── Ripple ───────────────────────────────────────────────────
function createRipple(e) {
  const btn  = e.currentTarget;
  const rect = btn.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height) * 2;
  const rip  = document.createElement('span');
  rip.className = 'ripple';
  rip.style.cssText = `width:${size}px;height:${size}px;left:${e.clientX-rect.left-size/2}px;top:${e.clientY-rect.top-size/2}px`;
  btn.appendChild(rip);
  rip.addEventListener('animationend', () => rip.remove());
}

// ── Helpers ──────────────────────────────────────────────────
function setBadge(type, text) {
  const el = document.getElementById('result-badge');
  el.className   = 'badge badge-' + type;
  el.textContent = text;
}

function animatePill(id, value) {
  const el = document.getElementById(id);
  el.classList.remove('pop');
  void el.offsetWidth;
  el.textContent = value;
  el.classList.add('pop');
}

function setFault(label) {
  const f    = FAULT_MAP[label];
  const icon = document.getElementById('fault-icon');
  const name = document.getElementById('fault-name');
  const code = document.getElementById('fault-code');
  icon.className   = 'fault-icon ' + (f ? f.cls : '');
  icon.textContent = f ? f.icon : '?';
  name.classList.remove('pop');
  void name.offsetWidth;
  name.textContent = f ? f.name : '—';
  name.classList.add('pop');
  code.textContent = f ? f.code : 'Unknown class';
}

function showSkeleton(show) {
  ['raw-skeleton', 'transform-skeleton'].forEach(id =>
    document.getElementById(id).classList.toggle('hidden', !show)
  );
}

function setPlotImg(id, b64) {
  const img = document.getElementById(id);
  img.classList.remove('loaded');
  img.src = 'data:image/png;base64,' + b64;
  img.onload = () => img.classList.add('loaded');
}

function showPlots(show) {
  document.getElementById('empty-state').classList.toggle('hidden', show);
  document.getElementById('plots-area').classList.toggle('visible', show);
}

function resetPlots() {
  ['raw-plot', 'transform-plot'].forEach(id => {
    const img = document.getElementById(id);
    img.src = ''; img.classList.remove('loaded');
  });
  document.getElementById('raw-meta').textContent       = '';
  document.getElementById('transform-meta').textContent = '';
}

// ── Run ──────────────────────────────────────────────────────
async function runDiagnosis() {
  if (isRunning) return;

  const modelPath  = MODEL_PATHS[currentTransform][currentModel];
  const signalPath = document.getElementById('signal-path').value.trim();
  const btn        = document.getElementById('run-btn');
  const statusLine = document.getElementById('status-line');

  if (!signalPath) {
    setBadge('error', 'Error');
    setFault(null);
    document.getElementById('fault-code').textContent = 'Please fill in the signal file path';
    animatePill('conf-val', '—');
    animatePill('time-val', '—');
    animatePill('idx-val',  '—');
    document.getElementById('conf-bar').style.width = '0%';
    document.getElementById('conf-pct-label').textContent = '';
    statusLine.className   = 'status-line error';
    statusLine.textContent = 'Signal path is required.';
    return;
  }

  // Loading
  isRunning    = true;
  btn.disabled = true;
  btn.querySelector('.btn-inner').innerHTML =
    `<span class="spinner"></span>Analyzing…`;

  setBadge('running', 'Running');
  document.getElementById('fault-icon').className   = 'fault-icon';
  document.getElementById('fault-icon').textContent = '…';
  document.getElementById('fault-name').textContent = 'Analyzing…';
  document.getElementById('fault-code').textContent = `${currentModel} · ${currentTransform.toUpperCase()}`;
  animatePill('conf-val', '—');
  animatePill('time-val', '—');
  animatePill('idx-val',  '—');
  document.getElementById('conf-bar').style.width = '0%';
  document.getElementById('conf-pct-label').textContent = '';
  statusLine.className   = 'status-line';
  statusLine.textContent = 'Sending to inference engine…';

  showPlots(true);
  showSkeleton(true);
  resetPlots();

  const t0 = performance.now();

  try {
    const response = await fetch('http://127.0.0.1:3636/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_path:  modelPath,
        signal_path: signalPath,
        transforms:  currentTransform,
      }),
    });

    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    const pct  = Math.round(data.confidence * 100);

    setBadge('success', 'Complete');
    setFault(data.label);
    animatePill('conf-val', pct + '%');
    animatePill('time-val', elapsed + 's');
    animatePill('idx-val',  data.segment_idx ?? '—');

    setTimeout(() => {
      document.getElementById('conf-bar').style.width = pct + '%';
      document.getElementById('conf-pct-label').textContent = pct + '%';
    }, 80);

    statusLine.className   = 'status-line';
    statusLine.textContent =
      `${currentModel} · ${currentTransform.toUpperCase()} · ${new Date().toLocaleTimeString()}`;

    showSkeleton(false);
    if (data.raw_plot && data.transform_plot) {
      const fileName = signalPath.split(/[\\/]/).pop();
      document.getElementById('raw-meta').textContent =
        `seg ${data.segment_idx ?? '?'} · ${fileName}`;
      document.getElementById('transform-meta').textContent =
        currentTransform.toUpperCase();
      setPlotImg('raw-plot',       data.raw_plot);
      setPlotImg('transform-plot', data.transform_plot);
    }

  } catch (err) {
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    showSkeleton(false);
    showPlots(false);
    setBadge('error', 'Error');
    document.getElementById('fault-icon').className   = 'fault-icon';
    document.getElementById('fault-icon').textContent = '✕';
    document.getElementById('fault-name').textContent = 'Diagnosis failed';
    document.getElementById('fault-code').textContent = '';
    animatePill('conf-val', '—');
    animatePill('time-val', elapsed + 's');
    animatePill('idx-val',  '—');
    document.getElementById('conf-bar').style.width = '0%';
    document.getElementById('conf-pct-label').textContent = '';
    statusLine.className   = 'status-line error';
    statusLine.textContent = err.message;

  } finally {
    isRunning    = false;
    btn.disabled = false;
    btn.querySelector('.btn-inner').innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="5 3 19 12 5 21 5 3"/>
      </svg>
      Run diagnosis`;
  }
}

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('run-btn').addEventListener('click', createRipple);
  document.getElementById('signal-path').addEventListener('keydown', e => {
    if (e.key === 'Enter') runDiagnosis();
  });
  document.getElementById('transform-sub').textContent = TRANSFORM_LABELS[currentTransform];
});