/* ============================================================
   Vibration Diagnosis — App.js
   SSE streaming: plots appear first, then each model result
   lands in place as it completes on the backend.
   ============================================================ */

const ARCHITECTURES = ['MobileNetV4', 'MobileOneS0', 'EdgeNeXtXXS', 'GhostNetV3'];
const TRANSFORMS    = ['scalogram', 'spectrogram'];

// Display names — order matches LABEL_CLASSES in inference.py.
// 0=Normal  1=IR  2=OR  3=B
const LABEL_NAMES = {
  0: 'Normal Condition',
  1: 'Outer Race Fault',
  2: 'Inner Race Fault',
  3: 'Ball Fault',
};

const fmtLabel = (n) => {
  if (n === null || n === undefined) return '—';
  // Already a readable string (e.g. ground truth passed as string)
  if (typeof n === 'string') return n;
  if (n < 0) return '—';
  return LABEL_NAMES[n] !== undefined ? LABEL_NAMES[n] : `Class ${n}`;
};

/* ------------------------------------------------------------------ */
/* DOM references                                                       */
/* ------------------------------------------------------------------ */
const el = {
  signalPath:       document.getElementById('signalPath'),
  runBtn:           document.getElementById('runBtn'),
  runBtnText:       document.querySelector('.run-btn-text'),
  connectionStatus: document.getElementById('connectionStatus'),
  statusText:       document.querySelector('#connectionStatus .status-text'),

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

/* ------------------------------------------------------------------ */
/* Build the 4-column x 2-row result grid once on page load            */
/* ------------------------------------------------------------------ */
function buildGrid() {
  el.modelsGrid.innerHTML = '';
  // Row 1: scalogram for all 4 archs, Row 2: spectrogram for all 4 archs
  TRANSFORMS.forEach(function(t) {
    ARCHITECTURES.forEach(function(arch) {
      var card = document.createElement('article');
      card.className = 'result-card waiting';
      card.setAttribute('data-arch', arch);
      card.setAttribute('data-transform', t);
      card.innerHTML =
        '<header class="result-head">' +
          '<span class="result-transform">' + (t === 'scalogram' ? 'Scalogram · CWT' : 'Spectrogram · STFT') + '</span>' +
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

/* ------------------------------------------------------------------ */
/* UI state helpers                                                     */
/* ------------------------------------------------------------------ */
function setRunning(running) {
  el.runBtn.disabled = running;
  el.runBtn.classList.toggle('running', running);
  el.runBtnText.textContent = running ? 'Running\u2026' : 'Run';
  el.connectionStatus.classList.remove('running', 'done', 'error');
  if (running) {
    el.connectionStatus.classList.add('running');
    el.statusText.textContent = 'Streaming';
    el.progressStrip.hidden = false;
    el.progressFill.style.width = '0%';
  }
}

function setChip(chip, label, cls) {
  chip.textContent = label;
  chip.className = 'chip ' + cls;
}

function markFrameLoading(frame, chip) {
  frame.classList.add('loading');
  frame.classList.remove('has-plot');
  setChip(chip, 'Computing', 'chip-loading');
}
function markFramePlot(frame, img, chip, dataUrl) {
  frame.classList.remove('loading');
  frame.classList.add('has-plot');
  img.src = dataUrl;
  setChip(chip, 'Ready', 'chip-done');
}
function markFrameError(frame, chip) {
  frame.classList.remove('loading');
  setChip(chip, 'Error', 'chip-error');
}

function resetAllFrames() {
  [
    [el.rawFrame,         el.rawImg,         el.rawChip],
    [el.spectrogramFrame, el.spectrogramImg, el.spectrogramChip],
    [el.scalogramFrame,   el.scalogramImg,   el.scalogramChip],
  ].forEach(function(arr) {
    var frame = arr[0], img = arr[1], chip = arr[2];
    frame.classList.remove('has-plot', 'loading');
    img.src = '';
    setChip(chip, 'Waiting', 'chip-idle');
  });
  el.gtPill.hidden = true;
}

function resetAllResultCards() {
  document.querySelectorAll('.result-card').forEach(function(card) {
    card.className = 'result-card waiting';
    card.querySelector('.result-badge').className = 'result-badge waiting';
    card.querySelector('.result-badge').textContent = 'Idle';
    card.querySelector('.result-body').innerHTML =
      '<div class="result-empty">' +
        '<span class="result-empty-dot"></span>' +
        '<span>Waiting for run</span>' +
      '</div>';
  });
}

function markCardLoading(arch, transform) {
  var card = document.querySelector(
    '.result-card[data-arch="' + arch + '"][data-transform="' + transform + '"]'
  );
  if (!card) return;
  card.className = 'result-card is-loading';
  var badge = card.querySelector('.result-badge');
  badge.className = 'result-badge loading';
  badge.textContent = 'Running';
  card.querySelector('.result-body').innerHTML =
    '<div class="result-empty">' +
      '<span class="result-empty-dot"></span>' +
      '<span>Inferring\u2026</span>' +
    '</div>';
}

function fillCard(result) {
  var card = document.querySelector(
    '.result-card[data-arch="' + result.model_name + '"][data-transform="' + result.transform + '"]'
  );
  if (!card) {
    console.warn('fillCard: no card for', result.model_name, result.transform);
    return;
  }

  if (result.error) {
    card.className = 'result-card error-card';
    var badge = card.querySelector('.result-badge');
    badge.className = 'result-badge incorrect';
    badge.textContent = 'Error';
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

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, function(c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}

/* ------------------------------------------------------------------ */
/* Progress bar                                                         */
/* ------------------------------------------------------------------ */
var TOTAL_STEPS = 3 + 8; // 3 plots + 8 model results
var completedSteps = 0;

function bumpProgress() {
  completedSteps++;
  el.progressFill.style.width = Math.min(100, (completedSteps / TOTAL_STEPS) * 100) + '%';
}

/* ------------------------------------------------------------------ */
/* Main run handler                                                     */
/* ------------------------------------------------------------------ */
async function run() {
  var signalPath = el.signalPath.value.trim();
  if (!signalPath) {
    el.signalPath.focus();
    el.signalPath.style.borderColor = 'var(--danger)';
    setTimeout(function() { el.signalPath.style.borderColor = ''; }, 1200);
    return;
  }

  completedSteps = 0;
  resetAllFrames();
  resetAllResultCards();
  setRunning(true);
  el.progressMessage.textContent = 'Connecting\u2026';

  markFrameLoading(el.rawFrame,         el.rawChip);
  markFrameLoading(el.spectrogramFrame, el.spectrogramChip);
  markFrameLoading(el.scalogramFrame,   el.scalogramChip);

  try {
    var resp = await fetch('/predict-stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ signal_path: signalPath }),
    });

    if (!resp.ok) {
      var txt = await resp.text();
      throw new Error('HTTP ' + resp.status + ': ' + txt.slice(0, 200));
    }
    if (!resp.body) {
      throw new Error('Browser does not support ReadableStream');
    }

    console.log('[run] stream connected, reading SSE...');
    await readSSE(resp.body.getReader(), handleEvent);
    console.log('[run] stream complete');

    el.connectionStatus.classList.remove('running');
    el.connectionStatus.classList.add('done');
    el.statusText.textContent = 'Complete';
    el.progressMessage.textContent = 'All models finished \u2713';
    el.progressFill.style.width = '100%';

  } catch (err) {
    console.error('[run] Fatal error:', err);
    el.connectionStatus.classList.remove('running');
    el.connectionStatus.classList.add('error');
    el.statusText.textContent = 'Error';
    el.progressMessage.textContent = 'Error: ' + err.message;
    markFrameError(el.rawFrame,         el.rawChip);
    markFrameError(el.spectrogramFrame, el.spectrogramChip);
    markFrameError(el.scalogramFrame,   el.scalogramChip);

  } finally {
    el.runBtn.disabled = false;
    el.runBtn.classList.remove('running');
    el.runBtnText.textContent = 'Run';
  }
}

/* ------------------------------------------------------------------ */
/* SSE parser                                                           */
/*                                                                      */
/* Named-event wire format the backend sends:                          */
/*   event: raw\n                                                       */
/*   data: {"plot":"data:image/png;base64,..."}\n                      */
/*   \n                                                                 */
/*                                                                      */
/* We buffer raw bytes, split on blank lines, parse each block.        */
/* ------------------------------------------------------------------ */
async function readSSE(reader, onEvent) {
  var decoder = new TextDecoder('utf-8');
  var buffer  = '';

  while (true) {
    var chunk = await reader.read();
    if (chunk.done) break;

    buffer += decoder.decode(chunk.value, { stream: true });

    // Process every complete message (terminated by \n\n)
    var boundary;
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      var block = buffer.slice(0, boundary);
      buffer    = buffer.slice(boundary + 2);
      if (block.trim()) parseSSEBlock(block, onEvent);
    }
  }

  // Flush any trailing data
  if (buffer.trim()) parseSSEBlock(buffer, onEvent);
}

function parseSSEBlock(block, onEvent) {
  // Normalise line endings
  var lines     = block.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
  var eventName = null;
  var dataStr   = null;

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      var piece = line.slice(5).trimStart();
      dataStr   = (dataStr === null) ? piece : dataStr + piece;
    }
  }

  if (!dataStr) {
    console.warn('[SSE] Block has no data line:', block.slice(0, 80));
    return;
  }

  var data;
  try {
    data = JSON.parse(dataStr);
  } catch (e) {
    console.error('[SSE] JSON.parse failed:', e.message);
    console.error('[SSE] data preview (first 300 chars):', dataStr.slice(0, 300));
    return;
  }

  // Named-event format (backend sends event: + data:)
  if (eventName) {
    console.debug('[SSE] event=' + eventName, Object.keys(data));
    onEvent({ event: eventName, data: data });
    return;
  }

  // Fallback: embedded format { "event": "...", "data": {...} }
  if (data.event && data.data !== undefined) {
    console.debug('[SSE] embedded event=' + data.event);
    onEvent(data);
    return;
  }

  console.warn('[SSE] Cannot determine event name from block:', block.slice(0, 100));
}

/* ------------------------------------------------------------------ */
/* Event router                                                         */
/* ------------------------------------------------------------------ */
function handleEvent(ev) {
  switch (ev.event) {

    case 'status':
      el.progressMessage.textContent = ev.data.message || '';
      break;

    case 'raw':
      markFramePlot(el.rawFrame, el.rawImg, el.rawChip, ev.data.plot);
      bumpProgress();
      break;

    case 'spectrogram':
      markFramePlot(el.spectrogramFrame, el.spectrogramImg, el.spectrogramChip, ev.data.plot);
      bumpProgress();
      break;

    case 'scalogram':
      markFramePlot(el.scalogramFrame, el.scalogramImg, el.scalogramChip, ev.data.plot);
      bumpProgress();
      break;

    case 'meta':
      el.gtValue.textContent = fmtLabel(ev.data.ground_truth);
      el.gtPill.hidden = false;
      ARCHITECTURES.forEach(function(arch) {
        TRANSFORMS.forEach(function(t) { markCardLoading(arch, t); });
      });
      break;

    case 'result':
      fillCard(ev.data);
      bumpProgress();
      break;

    case 'done':
      break;

    case 'error':
      console.error('[backend error]', ev.data.message);
      if (ev.data.traceback) console.error('[traceback]', ev.data.traceback);
      el.progressMessage.textContent = 'Error: ' + ev.data.message;
      el.progressStrip.style.background = 'rgba(255,59,48,0.08)';
      el.progressStrip.style.borderColor = 'rgba(255,59,48,0.25)';
      el.progressMessage.style.color = 'var(--danger)';
      el.connectionStatus.classList.remove('running');
      el.connectionStatus.classList.add('error');
      el.statusText.textContent = 'Error';
      // Only mark frames that haven't loaded yet as errors
      if (!el.rawFrame.classList.contains('has-plot'))
        markFrameError(el.rawFrame, el.rawChip);
      if (!el.spectrogramFrame.classList.contains('has-plot'))
        markFrameError(el.spectrogramFrame, el.spectrogramChip);
      if (!el.scalogramFrame.classList.contains('has-plot'))
        markFrameError(el.scalogramFrame, el.scalogramChip);
      break;

    default:
      console.warn('[SSE] Unknown event:', ev.event);
  }
}

/* ------------------------------------------------------------------ */
/* Event wiring                                                         */
/* ------------------------------------------------------------------ */
el.runBtn.addEventListener('click', run);
el.signalPath.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') run();
});