/* ============================================================
   Vibration Diagnosis — App.js  (non-streaming)
   Single POST /predict-all → all plots + 8 results at once.
   ============================================================ */

const ARCHITECTURES = ['MobileNetV4', 'TinyNetD', 'EdgeNeXtXXS', 'GhostNetV3'];
const TRANSFORMS    = ['scalogram', 'spectrogram'];

const LABEL_NAMES = {
  0: 'Normal Condition',
  1: 'Outer Race Fault',
  2: 'Inner Race Fault',
  3: 'Ball Fault',
};

function fmtLabel(n) {
  if (n === null || n === undefined) return '—';
  if (typeof n === 'string') return n;
  if (n < 0) return '—';
  return LABEL_NAMES[n] !== undefined ? LABEL_NAMES[n] : 'Class ' + n;
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, function(c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}

/* ── DOM refs ── */
const el = {
  signalPath:       document.getElementById('signalPath'),
  runBtn:           document.getElementById('runBtn'),
  runBtnText:       document.querySelector('.run-btn-text'),

  rawFrame:         document.getElementById('rawPlotFrame'),
  rawImg:           document.getElementById('rawPlot'),
  rawChip:          document.getElementById('rawChip'),

  spectrogramFrame: document.getElementById('spectrogramPlotFrame'),
  spectrogramImg:   document.getElementById('spectrogramPlot'),
  spectrogramChip:  document.getElementById('spectrogramChip'),

  scalogramFrame:   document.getElementById('scalogramPlotFrame'),
  scalogramImg:     document.getElementById('scalogramPlot'),
  scalogramChip:    document.getElementById('scalogramChip'),

  progressStrip:    document.getElementById('progressStrip'),
  progressMessage:  document.getElementById('progressMessage'),
  progressFill:     document.getElementById('progressFill'),

  modelsGrid:       document.getElementById('modelsGrid'),
  gtPill:           document.getElementById('groundTruthPill'),
  gtValue:          document.getElementById('groundTruthValue'),
};

/* ── Build 4×2 grid on load ── */
function buildGrid() {
  el.modelsGrid.innerHTML = '';
  TRANSFORMS.forEach(function(t) {
    ARCHITECTURES.forEach(function(arch) {
      var card = document.createElement('article');
      card.className = 'result-card waiting';
      card.setAttribute('data-arch', arch);
      card.setAttribute('data-transform', t);
      card.innerHTML =
        '<header class="result-head">' +
          '<span class="result-transform">' +
            (t === 'scalogram' ? 'Scalogram \xb7 CWT' : 'Spectrogram \xb7 STFT') +
          '</span>' +
          '<span class="result-badge waiting">Idle</span>' +
        '</header>' +
        '<div class="result-body">' +
          '<div class="result-empty">' +
            '<span class="result-empty-dot"></span>' +
            '<span>Waiting</span>' +
          '</div>' +
        '</div>';
      el.modelsGrid.appendChild(card);
    });
  });
}
buildGrid();

/* ── UI helpers ── */
function setChip(chip, label, cls) {
  chip.textContent = label;
  chip.className = 'chip ' + cls;
}

function setPlot(frame, img, chip, dataUrl) {
  frame.classList.remove('loading');
  frame.classList.add('has-plot');
  img.src = dataUrl;
  setChip(chip, 'Ready', 'chip-done');
}

function resetUI() {
  [
    [el.rawFrame,         el.rawImg,         el.rawChip],
    [el.spectrogramFrame, el.spectrogramImg, el.spectrogramChip],
    [el.scalogramFrame,   el.scalogramImg,   el.scalogramChip],
  ].forEach(function(arr) {
    arr[0].classList.remove('has-plot', 'loading');
    arr[1].src = '';
    setChip(arr[2], 'Waiting', 'chip-idle');
  });
  el.gtPill.hidden = true;
  el.progressStrip.hidden = true;
  el.progressFill.style.width = '0%';
  el.progressMessage.style.color = '';
  el.progressStrip.style.background = '';
  el.progressStrip.style.borderColor = '';
  buildGrid();
}

function setLoading(yes) {
  el.runBtn.disabled = yes;
  el.runBtn.classList.toggle('running', yes);
  el.runBtnText.textContent = yes ? 'Running\u2026' : 'Run';
  if (yes) {
    el.progressStrip.hidden = false;
    el.progressFill.style.width = '0%';
    el.progressMessage.textContent = 'Running inference\u2026';
    // Animate progress bar indeterminately while waiting
    el.progressFill.style.transition = 'width 8s linear';
    el.progressFill.style.width = '85%';
    // Mark all plot frames as loading
    [el.rawFrame, el.spectrogramFrame, el.scalogramFrame].forEach(function(f) {
      f.classList.add('loading');
    });
    setChip(el.rawChip,         'Computing', 'chip-loading');
    setChip(el.spectrogramChip, 'Computing', 'chip-loading');
    setChip(el.scalogramChip,   'Computing', 'chip-loading');
  } else {
    el.progressFill.style.transition = 'width 0.4s ease';
    el.progressFill.style.width = '100%';
    setTimeout(function() { el.progressMessage.textContent = 'Done \u2713'; }, 100);
  }
}

function fillCard(result) {
  var card = document.querySelector(
    '.result-card[data-arch="' + result.model_name + '"][data-transform="' + result.transform + '"]'
  );
  if (!card) return;

  if (result.error) {
    card.className = 'result-card error-card';
    card.querySelector('.result-badge').className = 'result-badge incorrect';
    card.querySelector('.result-badge').textContent = 'Error';
    card.querySelector('.result-body').innerHTML =
      '<div class="result-error-msg">' + escapeHtml(result.error) + '</div>';
    return;
  }

  card.className = 'result-card ' + (result.correct ? 'correct' : 'incorrect');
  var badge = card.querySelector('.result-badge');
  badge.className = 'result-badge ' + (result.correct ? 'correct' : 'incorrect');
  badge.textContent = result.correct ? '\u2713 Correct' : '\u2715 Wrong';

  var confPct = (result.confidence * 100).toFixed(1);
  card.querySelector('.result-body').innerHTML =
    '<div class="predicted-label">' +
      '<span class="predicted-label-eyebrow">Predicted</span>' +
      '<span class="predicted-label-value">' + escapeHtml(fmtLabel(result.label)) + '</span>' +
    '</div>' +
    '<div class="confidence-track">' +
      '<div class="confidence-fill" style="width:0%"></div>' +
    '</div>' +
    '<div class="result-stats">' +
      '<span class="result-stat-label">CONFIDENCE</span>' +
      '<span class="result-stat-value">' + confPct + '%</span>' +
    '</div>' +
    '<div class="result-stats">' +
      '<span class="result-stat-label">LATENCY</span>' +
      '<span class="result-stat-value">' + result.inference_time.toFixed(1) + ' ms</span>' +
    '</div>';

  requestAnimationFrame(function() {
    var fill = card.querySelector('.confidence-fill');
    if (fill) fill.style.width = confPct + '%';
  });
}

/* ── Main run ── */
async function run() {
  var signalPath = el.signalPath.value.trim();
  if (!signalPath) {
    el.signalPath.focus();
    el.signalPath.style.borderColor = 'var(--danger)';
    setTimeout(function() { el.signalPath.style.borderColor = ''; }, 1200);
    return;
  }

  resetUI();
  setLoading(true);

  try {
    var resp = await fetch('/predict-all', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signal_path: signalPath }),
    });

    if (!resp.ok) {
      var txt = await resp.text();
      var detail = txt;
      try { detail = JSON.parse(txt).detail || txt; } catch(e) {}
      throw new Error('HTTP ' + resp.status + ': ' + detail.slice(0, 200));
    }

    var data = await resp.json();

    /* ── Populate plots ── */
    setPlot(el.rawFrame,         el.rawImg,         el.rawChip,         data.raw_plot);
    setPlot(el.spectrogramFrame, el.spectrogramImg, el.spectrogramChip, data.spectrogram_plot);
    setPlot(el.scalogramFrame,   el.scalogramImg,   el.scalogramChip,   data.scalogram_plot);

    /* ── Plot timing ── */
    el.progressMessage.textContent = 'Plots rendered in ' + data.plot_time_ms.toFixed(1) + ' ms — running inference…';

    /* ── Ground truth ── */
    el.gtValue.textContent = fmtLabel(data.ground_truth);
    el.gtPill.hidden = false;

    /* ── Fill all 8 result cards ── */
    data.results.forEach(function(r) { fillCard(r); });

    el.progressMessage.textContent = 'Plots: ' + data.plot_time_ms.toFixed(1) + ' ms — all models done ✓';
    setLoading(false);

  } catch (err) {
    console.error('[run] error:', err);
    setLoading(false);
    el.progressFill.style.width = '100%';
    el.progressFill.style.background = 'var(--danger)';
    el.progressMessage.textContent = 'Error: ' + err.message;
    el.progressMessage.style.color = 'var(--danger)';
    el.progressStrip.style.background = 'rgba(255,59,48,0.06)';
    el.progressStrip.style.borderColor = 'rgba(255,59,48,0.2)';
    [el.rawFrame, el.spectrogramFrame, el.scalogramFrame].forEach(function(f) {
      f.classList.remove('loading');
    });
    setChip(el.rawChip,         'Error', 'chip-error');
    setChip(el.spectrogramChip, 'Error', 'chip-error');
    setChip(el.scalogramChip,   'Error', 'chip-error');
  }
}

el.runBtn.addEventListener('click', run);
el.signalPath.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') run();
});