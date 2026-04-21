/* ============================================================
   Vibration Monitor — monitor.js v3 (fixed)
   ============================================================ */

const LABEL_COLORS = {
  0: '#16a34a', // Normal   — dark green (visible on white)
  1: '#dc2626', // OR Fault — red
  2: '#ea580c', // IR Fault — orange-red
  3: '#d97706', // Ball     — amber
};
const FAULT_CLASS = { 0: 'normal', 1: 'fault', 2: 'fault', 3: 'fault' };

const $ = id => {
  const el = document.getElementById(id);
  if (!el) console.warn('[monitor] Missing DOM element:', id);
  return el || { hidden:false, textContent:'', innerHTML:'', className:'',
                 style:{}, dataset:{}, addEventListener:()=>{},
                 appendChild:()=>{}, querySelector:()=>null };
};
const el = {
  sigPath:         $('sigPath'),
  transformSel:    $('transformSel'),
  modelSel:        $('modelSel'),
  runBtn:          $('runBtn'),
  runTxt:          $('runTxt'),
  stopBtn:         $('stopBtn'),
  statusBadge:     $('statusBadge'),
  statusText:      $('statusText'),
  timerPill:       $('timerPill'),
  timerText:       $('timerText'),
  canvas:          $('sigCanvas'),
  faultBar:        $('faultBar'),
  gtBar:           $('gtBar'),
  gtBarRow:        $('gtBarRow'),
  faultTicks:      $('faultTicks'),
  thresholdSlider: $('thresholdSlider'),
  thresholdVal:    $('thresholdVal'),
  gtToggleWrap:    $('gtToggleWrap'),
  gtToggleBtn:     $('gtToggleBtn'),
  gtToggleTxt:     $('gtToggleTxt'),
  sumNormal:       $('sumNormal'),
  sumFault:        $('sumFault'),
  sumAvgPreproc:   $('sumAvgPreproc'),
  sumAvgModel:     $('sumAvgModel'),
  timeline:        $('timeline'),
  clearBtn:        $('clearBtn'),
};

/* ── Session state ── */
let abortCtrl    = null;
let timerInterval= null;
let startTime    = 0;
let preprocTimes = [];
let modelTimes   = [];
let nNormal      = 0;
let nFault       = 0;
let totalWindows = 0;
let hasGT        = false;
let gtWindowLabels = null;

/* ── Threshold slider ── */
function getThreshold() { return parseFloat(el.thresholdSlider.value); }
function updateSliderGradient(v) {
  el.thresholdSlider.style.background =
    `linear-gradient(90deg,#0a84ff ${v}%,rgba(60,60,67,.15) ${v}%)`;
}
el.thresholdSlider.addEventListener('input', () => {
  const v = getThreshold();
  el.thresholdVal.textContent = v + '%';
  updateSliderGradient(v);
});
updateSliderGradient(80);

/* ── GT bar visibility helper ── */
function setGTBarVisible(visible) {
  if (visible) {
    el.gtBarRow.removeAttribute('hidden');
  } else {
    el.gtBarRow.setAttribute('hidden', '');
  }
  el.gtToggleBtn.dataset.on  = String(visible);
  el.gtToggleTxt.textContent = visible ? 'Visible' : 'Hidden';
}

/* ── GT toggle — only works when GT data is loaded ── */
el.gtToggleBtn.addEventListener('click', () => {
  const isOn = el.gtToggleBtn.dataset.on === 'true';
  setGTBarVisible(!isOn);
});

/* ═══════════════════════════════════════════════════════════
   MATPLOTLIB-STYLE SIGNAL CANVAS
   White background, axes, grid, labeled ticks, scrolling waveform
   ═══════════════════════════════════════════════════════════ */
const ctx      = el.canvas.getContext('2d');
let oscBuf     = null;
let oscBufSize = 0;
let oscWrite   = 0;
let oscFilled  = 0;
let oscAmp     = 1;
let oscColor   = '#16a34a';
let rawSignal  = null;
let animFrame  = null;
let oscDirty   = false;
let sigDuration= 0;   // seconds — set from /signal-info

// Layout constants (in CSS px)
const PAD = { top: 18, right: 18, bottom: 36, left: 56 };

function initOscilloscope() {
  const rect = el.canvas.getBoundingClientRect();
  el.canvas.width  = rect.width  * devicePixelRatio;
  el.canvas.height = rect.height * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const W = rect.width;
  oscBufSize = Math.floor(W - PAD.left - PAD.right);
  oscBuf     = new Float32Array(oscBufSize);
  oscWrite   = 0; oscFilled = 0; oscAmp = 1;
  oscColor   = '#16a34a'; oscDirty = true;
  if (animFrame) cancelAnimationFrame(animFrame);
  animFrame = null;
  _rAF();
}

function _rAF() {
  animFrame = requestAnimationFrame(() => {
    if (oscDirty) { drawOscilloscope(); oscDirty = false; }
    _rAF();
  });
}

function pushSamples(samples) {
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    if (Math.abs(v) > oscAmp) oscAmp = Math.abs(v) * 1.05;
    oscBuf[oscWrite % oscBufSize] = v;
    oscWrite++;
    if (oscFilled < oscBufSize) oscFilled++;
  }
  oscDirty = true;
}

function drawOscilloscope() {
  const cssW = el.canvas.getBoundingClientRect().width;
  const cssH = el.canvas.getBoundingClientRect().height;

  // Plot area
  const px = PAD.left, py = PAD.top;
  const pw = cssW - PAD.left - PAD.right;
  const ph = cssH - PAD.top  - PAD.bottom;
  const mid = py + ph / 2;

  // ── Background ──────────────────────────────────────────
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, cssW, cssH);

  // Plot area white + border
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(px, py, pw, ph);

  // ── Grid lines ───────────────────────────────────────────
  const yGridCount = 4;
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth   = 1;
  ctx.setLineDash([3, 3]);
  for (let g = 0; g <= yGridCount; g++) {
    const y = py + (g / yGridCount) * ph;
    ctx.beginPath(); ctx.moveTo(px, y); ctx.lineTo(px + pw, y); ctx.stroke();
  }
  const xGridCount = 5;
  for (let g = 0; g <= xGridCount; g++) {
    const x = px + (g / xGridCount) * pw;
    ctx.beginPath(); ctx.moveTo(x, py); ctx.lineTo(x, py + ph); ctx.stroke();
  }
  ctx.setLineDash([]);

  // Centre zero-line
  ctx.strokeStyle = '#9ca3af';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(px, mid); ctx.lineTo(px + pw, mid); ctx.stroke();

  // ── Axes border ──────────────────────────────────────────
  ctx.strokeStyle = '#6b7280';
  ctx.lineWidth   = 1.5;
  ctx.strokeRect(px, py, pw, ph);

  // ── Y-axis labels (Amplitude) ────────────────────────────
  const amp = oscAmp || 1;
  ctx.fillStyle   = '#374151';
  ctx.font        = `${11 * devicePixelRatio / devicePixelRatio}px "JetBrains Mono", monospace`;
  ctx.textAlign   = 'right';
  ctx.textBaseline= 'middle';
  for (let g = 0; g <= yGridCount; g++) {
    const y   = py + (g / yGridCount) * ph;
    const val = amp - (g / yGridCount) * 2 * amp;
    ctx.fillText(val.toFixed(2), px - 6, y);
  }

  // Y-axis title
  ctx.save();
  ctx.translate(12, py + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign   = 'center';
  ctx.textBaseline= 'middle';
  ctx.fillStyle   = '#6b7280';
  ctx.font        = `11px "Inter Tight", sans-serif`;
  ctx.fillText('Amplitude', 0, 0);
  ctx.restore();

  // ── X-axis labels (Time) ──────────────────────────────────
  ctx.textAlign   = 'center';
  ctx.textBaseline= 'top';
  ctx.fillStyle   = '#374151';
  ctx.font        = `11px "JetBrains Mono", monospace`;
  const dur = sigDuration || 1;
  for (let g = 0; g <= xGridCount; g++) {
    const x = px + (g / xGridCount) * pw;
    const t = (oscFilled >= oscBufSize)
      ? (g / xGridCount) * dur
      : (g / xGridCount) * (oscFilled / oscBufSize) * dur;
    ctx.fillText(t.toFixed(1) + 's', x, py + ph + 5);
  }

  // X-axis title
  ctx.textAlign   = 'center';
  ctx.textBaseline= 'bottom';
  ctx.fillStyle   = '#6b7280';
  ctx.font        = `11px "Inter Tight", sans-serif`;
  ctx.fillText('Time (s)', px + pw / 2, cssH - 2);

  // ── Waveform ─────────────────────────────────────────────
  if (oscFilled < 2) return;
  const n      = Math.min(oscFilled, oscBufSize);
  const yScale = (ph * 0.44) / amp;

  ctx.save();
  ctx.beginPath();
  ctx.rect(px, py, pw, ph);
  ctx.clip();

  // Subtle fill under curve
  const grad = ctx.createLinearGradient(0, mid - ph * 0.4, 0, mid + ph * 0.4);
  grad.addColorStop(0, oscColor + '18');
  grad.addColorStop(1, oscColor + '04');
  ctx.fillStyle = grad;
  ctx.lineWidth  = 0;
  ctx.beginPath();
  const startIdx = oscFilled >= oscBufSize ? oscWrite : 0;
  ctx.moveTo(px, mid);
  for (let i = 0; i < n; i++) {
    const idx = (startIdx + i) % oscBufSize;
    const x   = px + (i / (oscBufSize - 1)) * pw;
    const y   = mid - oscBuf[idx] * yScale;
    i === 0 ? ctx.lineTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.lineTo(px + (n - 1) / (oscBufSize - 1) * pw, mid);
  ctx.closePath();
  ctx.fill();

  // Signal line
  ctx.strokeStyle = oscColor;
  ctx.lineWidth   = 1.6;
  ctx.lineJoin    = 'round';
  ctx.lineCap     = 'round';
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const idx = (startIdx + i) % oscBufSize;
    const x   = px + (i / (oscBufSize - 1)) * pw;
    const y   = mid - oscBuf[idx] * yScale;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function stopOscilloscope() {
  if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
}

/* ── Fault bar ── */
function addFaultSegment(result) {
  if (!totalWindows) return;
  const pct   = (1 / totalWindows) * 100;
  const color = LABEL_COLORS[result.label] || '#8e8e93';
  const seg   = document.createElement('div');
  seg.title   = `t=${result.t_start.toFixed(2)}s: ${result.label_name} (${result.confidence}%)`;
  seg.style.cssText = `position:absolute;left:${result.window_idx/totalWindows*100}%;width:${pct}%;height:100%;background:${color};opacity:0.85;transition:opacity .15s;`;
  seg.addEventListener('mouseenter', () => seg.style.opacity = '1');
  seg.addEventListener('mouseleave', () => seg.style.opacity = '0.85');
  el.faultBar.appendChild(seg);
}

/* ── GT bar ── */
function buildGTBar(gtLabels) {
  el.gtBar.innerHTML = '';
  const n = gtLabels.length;
  for (let i = 0; i < n; i++) {
    const color = LABEL_COLORS[gtLabels[i]] || '#8e8e93';
    const seg   = document.createElement('div');
    seg.title   = `Window ${i}: ${['Normal','OR Fault','IR Fault','Ball Fault'][gtLabels[i]] ?? gtLabels[i]}`;
    seg.style.cssText = `position:absolute;left:${i/n*100}%;width:${1/n*100}%;height:100%;background:${color};opacity:0.85;`;
    el.gtBar.appendChild(seg);
  }
}

function buildFaultTicks(durationS) {
  el.faultTicks.innerHTML = '';
  const steps = Math.min(10, Math.floor(durationS));
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * durationS;
    const d = document.createElement('span');
    d.textContent = t.toFixed(1) + 's';
    el.faultTicks.appendChild(d);
  }
}

/* ── Timeline ── */
function addTimelineEntry(r) {
  const empty = el.timeline.querySelector('.timeline-empty');
  if (empty) empty.remove();
  const cls   = FAULT_CLASS[r.label] || 'fault';
  const item  = document.createElement('div');
  item.className = 'tl-item ' + cls + (r.below_threshold ? ' below-thresh' : '');
  item.innerHTML =
    `<span class="tl-dot ${cls}"></span>` +
    `<span class="tl-time">${r.t_start.toFixed(3)}s</span>` +
    `<span class="tl-name ${cls}">${r.label_name}` +
      (r.below_threshold ? ' <em class="tl-low">(low)</em>' : '') +
    `</span>` +
    `<span class="tl-conf">${r.confidence}%</span>` +
    `<span class="tl-preproc">${r.preproc_ms.toFixed(1)}ms</span>` +
    `<span class="tl-sep">+</span>` +
    `<span class="tl-model">${r.model_ms.toFixed(1)}ms</span>`;
  el.timeline.appendChild(item);
  el.timeline.scrollTop = el.timeline.scrollHeight;
}

/* ── Summary ── */
function updateSummary(r) {
  if (r.label === 0) nNormal++; else nFault++;
  preprocTimes.push(r.preproc_ms);
  modelTimes.push(r.model_ms);
  const avgP = preprocTimes.reduce((a,b)=>a+b,0)/preprocTimes.length;
  const avgM = modelTimes.reduce((a,b)=>a+b,0)/modelTimes.length;
  el.sumNormal.textContent     = nNormal;
  el.sumFault.textContent      = nFault;
  el.sumAvgPreproc.textContent = avgP.toFixed(1) + 'ms';
  el.sumAvgModel.textContent   = avgM.toFixed(1) + 'ms';
}

/* ── Status / Timer ── */
function setStatus(state, text) {
  el.statusBadge.className  = 'badge ' + state;
  el.statusText.textContent = text;
}
function startTimer() {
  startTime = Date.now(); el.timerPill.hidden = false;
  timerInterval = setInterval(() => {
    el.timerText.textContent = ((Date.now()-startTime)/1000).toFixed(2)+' s';
  }, 100);
}
function stopTimer() { clearInterval(timerInterval); }

/* ── Reset ── */
function resetUI() {
  nNormal=0; nFault=0; preprocTimes=[]; modelTimes=[];
  totalWindows=0; hasGT=false; gtWindowLabels=null; rawSignal=null;
  oscAmp=1; sigDuration=0;
  el.faultBar.innerHTML=''; el.gtBar.innerHTML=''; el.faultTicks.innerHTML='';
  // Always hide GT bar and reset toggle state on reset
  setGTBarVisible(false);
  el.gtToggleWrap.hidden=true;
  el.timerPill.hidden=true;
  el.sumNormal.textContent='0'; el.sumFault.textContent='0';
  el.sumAvgPreproc.textContent='—'; el.sumAvgModel.textContent='—';
  el.timeline.innerHTML='<div class="timeline-empty">No predictions yet</div>';
}

/* ── Main run ── */
async function run() {
  const path      = el.sigPath.value.trim();
  const transform = el.transformSel.value;
  const model     = el.modelSel.value;
  const threshold = getThreshold();
  if (!path) { el.sigPath.focus(); return; }

  resetUI();
  stopOscilloscope();
  initOscilloscope();

  el.runBtn.disabled=true; el.runBtn.classList.add('running');
  el.runTxt.textContent='Loading…'; el.stopBtn.hidden=false;
  abortCtrl = new AbortController();

  try {
    setStatus('running','Loading…');

    const infoResp = await fetch('/signal-info', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({signal_path:path, model_name:model, transform, threshold}),
      signal: abortCtrl.signal,
    });
    if (!infoResp.ok) throw new Error('signal-info: HTTP '+infoResp.status);
    const info = await infoResp.json();

    rawSignal      = info.waveform;
    totalWindows   = info.total_windows;
    sigDuration    = info.duration_s;
    hasGT          = info.has_ground_truth;
    gtWindowLabels = info.gt_window_labels;

    buildFaultTicks(info.duration_s);
    pushSamples(rawSignal);  // pre-fill canvas
    oscColor = '#6b7280';    // neutral gray before first prediction

    if (hasGT && gtWindowLabels) {
      buildGTBar(gtWindowLabels);
      // Show the toggle widget but keep the bar hidden until user clicks ON
      el.gtToggleWrap.hidden = false;
      setGTBarVisible(false); // explicitly ensure bar is hidden
    }

    setStatus('running','Streaming…');
    el.runTxt.textContent='Running…';
    startTimer();

    const resp = await fetch('/monitor-stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({signal_path:path, model_name:model, transform, threshold}),
      signal: abortCtrl.signal,
    });
    if (!resp.ok) throw new Error('monitor-stream: HTTP '+resp.status);

    await readSSE(resp.body.getReader(), (event, data) => {
      if (event === 'window') {
        const step    = info.waveform_step;
        const dsStart = Math.floor(data.start_sample / step);
        const dsEnd   = Math.floor(data.end_sample   / step);
        oscColor = LABEL_COLORS[data.label] || '#16a34a';
        pushSamples(rawSignal.slice(dsStart, dsEnd));
        addTimelineEntry(data);
        addFaultSegment(data);
        updateSummary(data);
      } else if (event === 'done') {
        finish('done','Complete');
      } else if (event === 'error') {
        throw new Error(data.message);
      }
    });

  } catch(err) {
    if (err.name==='AbortError') finish('done','Stopped');
    else { console.error(err); setStatus('error','Error'); finish('error','Error'); }
  }
}

function finish(state, label) {
  stopTimer(); setStatus(state,label);
  el.runBtn.disabled=false; el.runBtn.classList.remove('running');
  el.runTxt.textContent='Run'; el.stopBtn.hidden=true;
}
function stop() { if (abortCtrl) abortCtrl.abort(); }

/* ── SSE reader ── */
async function readSSE(reader, onEvent) {
  const dec = new TextDecoder('utf-8');
  let buf = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream:true});
    let b;
    while ((b = buf.indexOf('\n\n')) !== -1) {
      const block = buf.slice(0, b); buf = buf.slice(b+2);
      if (!block.trim()) continue;
      let name=null, data=null;
      for (const line of block.replace(/\r\n/g,'\n').split('\n')) {
        if (line.startsWith('event:'))    name = line.slice(6).trim();
        else if (line.startsWith('data:')) data = (data||'')+line.slice(5).trimStart();
      }
      if (!data) continue;
      try { onEvent(name, JSON.parse(data)); } catch(e) {}
    }
  }
}

/* ── Wiring ── */
el.runBtn.addEventListener('click', run);
el.stopBtn.addEventListener('click', stop);
el.clearBtn.addEventListener('click', () => {
  el.timeline.innerHTML='<div class="timeline-empty">No predictions yet</div>';
});
el.sigPath.addEventListener('keydown', e => { if (e.key==='Enter') run(); });
window.addEventListener('resize', () => { if (oscBuf) { stopOscilloscope(); initOscilloscope(); } });