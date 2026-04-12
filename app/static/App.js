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

let currentTransform = 'cwt';
let currentModel = 'MobileOneS0';
let isRunning = false;

// ── Transform switch ──────────────────────────────────────────
function switchTransform(t) {
  if (currentTransform === t) return;
  currentTransform = t;
  document.getElementById('tab-cwt').classList.toggle('active', t === 'cwt');
  document.getElementById('tab-stft').classList.toggle('active', t === 'stft');
}

// ── Model select ──────────────────────────────────────────────
function selectModel(m) {
  if (currentModel === m) return;
  currentModel = m;
  MODEL_KEYS.forEach(name => {
    document.getElementById('btn-' + name).classList.toggle('active', name === m);
  });
}

// ── Ripple effect ─────────────────────────────────────────────
function createRipple(e) {
  const btn = e.currentTarget;
  const rect = btn.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height) * 2;
  const x = e.clientX - rect.left - size / 2;
  const y = e.clientY - rect.top - size / 2;
  const ripple = document.createElement('span');
  ripple.className = 'ripple';
  ripple.style.cssText = `width:${size}px;height:${size}px;left:${x}px;top:${y}px`;
  btn.appendChild(ripple);
  ripple.addEventListener('animationend', () => ripple.remove());
}

// ── Waveform control ──────────────────────────────────────────
function setWaveform(active) {
  document.getElementById('waveform').classList.toggle('idle', !active);
}

// ── Set badge ─────────────────────────────────────────────────
function setBadge(type, text) {
  const badge = document.getElementById('result-badge');
  badge.className = 'badge badge-' + type;
  badge.textContent = text;
}

// ── Animate number ────────────────────────────────────────────
function animateValue(el, value) {
  el.classList.remove('pop');
  void el.offsetWidth;
  el.textContent = value;
  el.classList.add('pop');
}

// ── Run diagnosis ─────────────────────────────────────────────
async function runDiagnosis() {
  if (isRunning) return;

  const modelPath  = MODEL_PATHS[currentTransform][currentModel];
  const signalPath = document.getElementById('signal-path').value.trim();
  const btn        = document.getElementById('run-btn');
  const resultCard = document.getElementById('result-card');
  const statusLine = document.getElementById('status-line');
  const labelVal   = document.getElementById('label-val');
  const confVal    = document.getElementById('conf-val');
  const confBar    = document.getElementById('conf-bar');
  const confPct    = document.getElementById('conf-pct');

  // Validation
  if (!signalPath) {
    resultCard.classList.add('visible', 'error');
    resultCard.classList.remove('success');
    setBadge('error', 'Error');
    animateValue(labelVal, '—');
    animateValue(confVal, '—');
    confBar.style.width = '0%';
    confPct.textContent = '';
    statusLine.className = 'status-line error';
    statusLine.textContent = 'Please fill in the signal file path.';
    setWaveform(false);
    return;
  }

  // Loading state
  isRunning = true;
  btn.disabled = true;
  btn.querySelector('.btn-inner').innerHTML = `<span class="spinner"></span>Analyzing…`;
  resultCard.classList.add('visible');
  resultCard.classList.remove('success', 'error');
  setBadge('pending', 'Running');
  animateValue(labelVal, '—');
  animateValue(confVal, '—');
  confBar.style.width = '0%';
  confPct.textContent = '';
  statusLine.className = 'status-line';
  statusLine.textContent = 'Sending to inference engine…';
  setWaveform(true);

  try {
    const response = await fetch('http://127.0.0.1:3636/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath, signal_path: signalPath, transforms: currentTransform }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    const pct  = Math.round(data.confidence * 100);

    // Success
    resultCard.classList.add('success');
    setBadge('success', 'Complete');
    animateValue(labelVal, data.label);
    animateValue(confVal, pct + '%');

    setTimeout(() => {
      confBar.style.width = pct + '%';
      confPct.textContent = pct + '%';
    }, 80);

    statusLine.className = 'status-line';
    statusLine.textContent = `${currentModel} · ${currentTransform.toUpperCase()} · ${new Date().toLocaleTimeString()}`;
    setWaveform(false);

  } catch (err) {
    resultCard.classList.add('error');
    setBadge('error', 'Error');
    animateValue(labelVal, '—');
    animateValue(confVal, '—');
    confBar.style.width = '0%';
    confPct.textContent = '';
    statusLine.className = 'status-line error';
    statusLine.textContent = err.message;
    setWaveform(false);
  } finally {
    isRunning = false;
    btn.disabled = false;
    btn.querySelector('.btn-inner').innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
      </svg>
      Run diagnosis`;
  }
}

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('run-btn').addEventListener('click', createRipple);

  document.getElementById('signal-path').addEventListener('keydown', e => {
    if (e.key === 'Enter') runDiagnosis();
  });
});