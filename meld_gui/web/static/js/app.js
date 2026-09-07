/* MELD Detector — application controller. */
(function () {
  'use strict';

  const { $, $$, el, escapeHtml, fmt } = window.U;
  const U = window.U;

  const State = {
    config: null,
    model: null,
    result: null,
    text: '',
    sentSort: { key: 'index', dir: 'asc' },
    batchFiles: [],
    batchJob: null,
    batchTimer: null,
    modelTimer: null,
    history: { offset: 0, limit: 25, q: '', flagged: false, total: 0 }
  };

  const SAMPLE = [
    'The integration of renewable energy sources into existing power grids represents one of the ',
    'most significant infrastructure challenges of the coming decade. It is important to note that ',
    'this transition involves far more than simply installing additional generation capacity. ',
    'Furthermore, grid operators must contend with the intermittent nature of solar and wind ',
    'resources, which introduces variability that traditional baseload plants were never designed ',
    'to accommodate.\n\nSeveral key strategies have emerged to address these challenges. First, ',
    'utility-scale battery storage can absorb excess generation during peak production periods and ',
    'release it when demand outstrips supply. Second, demand response programs incentivise ',
    'consumers to shift consumption to periods of abundant generation. Finally, improved ',
    'interconnection between regional grids allows surplus power to flow toward areas of shortfall, ',
    'effectively smoothing regional variability across a wider geographic footprint.\n\n',
    'In conclusion, while the technical obstacles are substantial, they are not insurmountable. ',
    'A combination of storage, flexible demand and enhanced transmission infrastructure offers a ',
    'viable pathway forward for grid operators seeking to decarbonise their generation portfolios.'
  ].join('');

  /* ==================================================================== init */
  document.addEventListener('DOMContentLoaded', async function () {
    initTheme();
    bindNav();
    bindTabs();
    bindEditor();
    bindAnalyze();
    bindBatch();
    bindCompare();
    bindHistory();
    bindModelPanel();
    renderGuide();

    try {
      State.config = await window.API.config();
      populateSelects();
    } catch (err) {
      U.toast('Could not load configuration: ' + err.message, 'error');
    }

    pollModel();
    updateCounts();
  });

  /* =================================================================== theme */
  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem('meld-theme'); } catch (_) { /* blocked */ }
    if (saved) document.documentElement.setAttribute('data-theme', saved);

    $('#themeToggle').addEventListener('click', function () {
      const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('meld-theme', next); } catch (_) { /* blocked */ }
      if (State.result) renderResult(State.result);   // recolour charts
    });
  }

  /* ============================================================== navigation */
  function bindNav() {
    $$('.nav-item').forEach(function (btn) {
      btn.addEventListener('click', function () {
        const view = btn.dataset.view;
        $$('.nav-item').forEach((b) => b.classList.toggle('is-active', b === btn));
        $$('.view').forEach((v) => v.classList.toggle('is-active', v.dataset.view === view));
        if (view === 'history') loadHistory();
        if (view === 'model') refreshModelPanel();
      });
    });
  }

  function bindTabs() {
    $$('.tab').forEach(function (tab) {
      tab.addEventListener('click', function () {
        const name = tab.dataset.tab;
        $$('.tab').forEach((t) => t.classList.toggle('is-active', t === tab));
        $$('.tab-panel').forEach((p) => p.classList.toggle('is-active', p.dataset.tab === name));
      });
    });
  }

  /* ================================================================ selects */
  function populateSelects() {
    const strata = State.config.strata;
    const fprs = State.config.fpr_levels;

    ['#selStratum', '#batchStratum', '#cmpStratum'].forEach(function (sel) {
      const node = $(sel);
      node.innerHTML = '';
      strata.forEach((s) => node.appendChild(el('option', { value: s.key, text: s.label })));
    });
    ['#selFpr', '#batchFpr', '#cmpFpr'].forEach(function (sel) {
      const node = $(sel);
      node.innerHTML = '';
      fprs.forEach((f) => node.appendChild(el('option', { value: f.key, text: f.label })));
    });

    $('#dropTypes').textContent = State.config.supported_files
      .map((s) => s.replace('.', '')).join(' · ');

    $('#selStratum').addEventListener('change', updateStrataHint);
    updateStrataHint();
  }

  function updateStrataHint() {
    const key = $('#selStratum').value;
    const entry = (State.config.strata || []).find((s) => s.key === key);
    $('#strataHint').textContent = entry
      ? entry.hint + ' Thresholds are calibrated separately for each type.'
      : '';
  }

  /* ================================================================= editor */
  function bindEditor() {
    const editor = $('#editor');
    editor.addEventListener('input', U.debounce(updateCounts, 120));

    $('#btnSample').addEventListener('click', function () {
      editor.value = SAMPLE;
      updateCounts();
      U.toast('Sample loaded — a synthetic, AI-style passage for trying the tool.', 'info');
    });

    $('#btnClear').addEventListener('click', function () {
      editor.value = '';
      updateCounts();
      State.result = null;
      $('#resultBody').hidden = true;
      $('#resultEmpty').hidden = false;
    });

    // Drag and drop onto the editor.
    const zone = $('#dropZone');
    ['dragenter', 'dragover'].forEach((evt) =>
      zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add('is-dragging'); }));
    ['dragleave', 'drop'].forEach((evt) =>
      zone.addEventListener(evt, (e) => {
        e.preventDefault();
        if (evt === 'dragleave' && zone.contains(e.relatedTarget)) return;
        zone.classList.remove('is-dragging');
      }));
    zone.addEventListener('drop', function (e) {
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) loadFileIntoEditor(file);
    });

    $('#btnUpload').addEventListener('click', () => $('#fileInput').click());
    $('#fileInput').addEventListener('change', function (e) {
      if (e.target.files[0]) loadFileIntoEditor(e.target.files[0]);
      e.target.value = '';
    });

    document.addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); runAnalyze(); }
    });
  }

  async function loadFileIntoEditor(file) {
    U.busy(true, 'Extracting text from ' + file.name + '…');
    try {
      const data = await window.API.extract(file);
      $('#editor').value = data.text;
      updateCounts();
      U.toast('Loaded ' + data.name + ' — ' + fmt.int(data.statistics.words) + ' words', 'success');
    } catch (err) {
      U.toast(err.message, 'error');
    } finally {
      U.busy(false);
    }
  }

  function updateCounts() {
    const text = $('#editor').value;
    const words = (text.match(/\b[\w'’-]+\b/g) || []).length;
    const min = (State.config && State.config.min_words) || 100;

    $('#wcWords').textContent = fmt.int(words) + ' words';
    $('#wcChars').textContent = fmt.int(text.length) + ' chars';
    $('#wcTokens').textContent = '~' + fmt.int(Math.round(words * 1.33)) + ' tokens';

    // The bar fills to the 100-word floor at 25%, then tracks toward 2,048 tokens.
    const fill = $('#lengthFill');
    const ratio = words <= min
      ? (words / min) * 25
      : 25 + Math.min(75, ((words - min) / (1500 - min)) * 75);
    fill.style.width = Math.min(100, ratio) + '%';
    fill.classList.toggle('is-short', words < min);

    $('#lengthHint').textContent = words === 0
      ? 'Below ' + min + ' words MELD is unreliable.'
      : words < min
        ? words + ' words — ' + (min - words) + ' short of the ' + min + '-word reliability floor.'
        : words + ' words — above the reliability floor.';
    $('#lengthHint').style.color = words && words < min ? U.cssVar('--uncertain') : '';
  }

  /* =============================================================== analysis */
  function bindAnalyze() {
    $('#btnAnalyze').addEventListener('click', runAnalyze);
    $('#chkTokenHeat').addEventListener('change', () => State.result && renderHeatmap(State.result));
    $('#sentSearch').addEventListener('input', U.debounce(renderSentences, 150));
    $('#chkFlaggedOnly').addEventListener('change', renderSentences);

    $$('#sentTable th[data-sort]').forEach(function (th) {
      th.addEventListener('click', function () {
        const key = th.dataset.sort;
        const s = State.sentSort;
        s.dir = s.key === key && s.dir === 'asc' ? 'desc' : 'asc';
        s.key = key;
        renderSentences();
      });
    });

    $('#btnCopyJson').addEventListener('click', function () {
      navigator.clipboard.writeText(JSON.stringify(State.result, null, 2))
        .then(() => U.toast('JSON copied to clipboard', 'success'))
        .catch(() => U.toast('Clipboard blocked by the browser', 'error'));
    });
    $('#btnDownloadJson').addEventListener('click', function () {
      U.download('meld-analysis.json', JSON.stringify(State.result, null, 2), 'application/json');
    });
    $('#btnDownloadCsv').addEventListener('click', function () {
      if (!State.result || !State.result.sentences) return;
      const cols = ['index', 'score', 'probability', 'flagged', 'words', 'tokens', 'text'];
      U.download('meld-sentences.csv', U.toCsv(State.result.sentences, cols), 'text/csv');
    });
    $('#btnPrint').addEventListener('click', () => window.print());
  }

  async function runAnalyze() {
    const text = $('#editor').value;
    if (!text.trim()) { U.toast('Paste or load a document first.', 'error'); return; }
    if (!State.model || !State.model.loaded) {
      U.toast('The model is not loaded yet — open the Model tab to load it.', 'error');
      return;
    }

    U.busy(true, 'Scoring document…');
    try {
      const result = await window.API.analyze({
        text: text,
        stratum: $('#selStratum').value,
        fpr: $('#selFpr').value,
        include_tokens: true,
        include_sentences: true,
        include_attribution: true,
        save_history: true
      });
      State.text = text;
      State.result = result;
      renderResult(result);
      U.toast('Scored in ' + fmt.duration(result.timing.seconds)
        + ' on ' + result.timing.device.toUpperCase(), 'success');
    } catch (err) {
      U.toast(err.message, 'error');
    } finally {
      U.busy(false);
    }
  }

  /* ============================================================== rendering */
  function renderResult(result) {
    $('#resultEmpty').hidden = true;
    $('#resultBody').hidden = false;

    const d = result.decision;
    const color = U.verdictColor(d.verdict);
    const card = $('#verdictCard');
    card.style.setProperty('--verdict-color', color);
    card.style.setProperty('--verdict-soft', U.verdictSoft(d.verdict));

    $('#gaugeValue').textContent = fmt.pct(d.probability, 1);
    window.Charts.gauge($('#gauge'), d.probability, d.threshold, color);

    $('#verdictBadge').textContent = d.verdict_label;
    $('#verdictText').innerHTML = verdictSentence(d, result);
    window.Charts.scoreScale($('#scoreScale'), d.score, d.threshold);

    $('#metrics').innerHTML = [
      metric('Raw score', fmt.signed(d.score)),
      metric('Cut-off', fmt.signed(d.threshold)),
      metric('Margin', fmt.signed(d.margin)),
      metric('Words', fmt.int(result.statistics.words)),
      metric('Windows', String(result.tokens.windows))
    ].join('');

    renderAlerts(result);
    renderHeatmap(result);
    renderSentences();
    renderAttribution(result);
    renderWindows(result);
    renderStats(result);
    $('#jsonView').innerHTML = U.highlightJson(result);
  }

  function metric(label, value) {
    return '<div class="metric"><div class="metric-label">' + escapeHtml(label)
      + '</div><div class="metric-value">' + escapeHtml(value) + '</div></div>';
  }

  function verdictSentence(d, result) {
    const above = d.flagged;
    return 'Raw score <b>' + fmt.signed(d.score) + '</b> is '
      + (above ? '<b>above</b>' : 'below')
      + ' the <b>' + fmt.signed(d.threshold) + '</b> cut-off for <b>'
      + escapeHtml(d.stratum_label.toLowerCase()) + '</b> at a '
      + d.fpr_pct + '% false-positive rate. '
      + (above
        ? 'Roughly ' + d.fpr_pct + ' in 100 genuinely human documents of this type would also land here.'
        : 'This document sits inside the human range for these settings.');
  }

  function renderAlerts(result) {
    const host = $('#alerts');
    host.innerHTML = '';
    (result.warnings || []).forEach(function (w) {
      const icon = w.level === 'warning'
        ? '<svg viewBox="0 0 20 20"><path d="M10 3l7.5 13H2.5L10 3z"/><path d="M10 8v3.4M10 13.6v.4"/></svg>'
        : '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7.5"/><path d="M10 9v5M10 6.2v.6"/></svg>';
      host.insertAdjacentHTML('beforeend',
        '<div class="alert alert-' + w.level + '">' + icon
        + '<div>' + escapeHtml(w.message) + '</div></div>');
    });
  }

  /* ------------------------------------------------------------- heatmap -- */
  function renderHeatmap(result) {
    const host = $('#heatmap');
    const text = State.text || '';
    const threshold = result.decision.threshold;
    const tokenMode = $('#chkTokenHeat').checked;

    $('#heatLegend').innerHTML =
      '<span>Human</span><div class="legend-scale"></div><span>Machine</span>'
      + '<span style="margin-left:auto">colour = margin against the '
      + fmt.signed(threshold) + ' cut-off</span>';

    const spans = tokenMode && result.token_detail
      ? result.token_detail.offsets.map((o, i) => ({
          start: o[0], end: o[1], score: result.token_detail.scores[i], token: true
        }))
      : (result.sentences || []).map((s) => ({
          start: s.start, end: s.end, score: s.score, sentence: s
        }));

    if (!spans.length) {
      host.innerHTML = '<p class="hint">No spans to display.</p>';
      return;
    }

    let html = '';
    let cursor = 0;
    spans.forEach(function (span) {
      if (span.start > cursor) html += escapeHtml(text.slice(cursor, span.start));
      if (span.end <= span.start) return;
      const margin = span.score - threshold;
      const body = escapeHtml(text.slice(span.start, span.end));
      const title = span.sentence
        ? 'score ' + fmt.signed(span.sentence.score)
          + ' · P(AI) ' + fmt.pct(span.sentence.probability)
          + ' · ' + span.sentence.tokens + ' tokens'
        : 'token score ' + fmt.signed(span.score);
      html += '<span class="heat-seg' + (span.token ? ' heat-tok' : '') + '"'
        + ' style="background:' + U.heatColor(margin, span.token ? 0.30 : 0.22) + '"'
        + ' title="' + escapeHtml(title) + '">' + body + '</span>';
      cursor = Math.max(cursor, span.end);
    });
    if (cursor < text.length) html += escapeHtml(text.slice(cursor));

    host.innerHTML = html;
  }

  /* ----------------------------------------------------------- sentences -- */
  function renderSentences() {
    const result = State.result;
    const body = $('#sentTable tbody');
    if (!result || !result.sentences) { body.innerHTML = ''; return; }

    const query = $('#sentSearch').value.toLowerCase();
    const flaggedOnly = $('#chkFlaggedOnly').checked;
    const { key, dir } = State.sentSort;

    const rows = result.sentences
      .filter((s) => (!flaggedOnly || s.flagged)
        && (!query || s.text.toLowerCase().includes(query)))
      .sort(function (a, b) {
        const x = a[key], y = b[key];
        const cmp = typeof x === 'string' ? x.localeCompare(y) : x - y;
        return dir === 'asc' ? cmp : -cmp;
      });

    $$('#sentTable th[data-sort]').forEach(function (th) {
      th.classList.remove('sorted-asc', 'sorted-desc');
      if (th.dataset.sort === key) th.classList.add('sorted-' + dir);
    });

    const threshold = result.decision.threshold;
    body.innerHTML = rows.map(function (s) {
      const color = U.heatColor(s.score - threshold);
      return '<tr>'
        + '<td class="col-num muted">' + (s.index + 1) + '</td>'
        + '<td class="col-num" style="color:' + color + '">' + fmt.signed(s.score) + '</td>'
        + '<td class="col-num">' + fmt.pct(s.probability, 1) + '</td>'
        + '<td class="col-num muted">' + s.words + '</td>'
        + '<td class="cell-text">' + escapeHtml(s.text.slice(0, 260))
        + (s.text.length > 260 ? '…' : '') + '</td>'
        + '</tr>';
    }).join('') || '<tr><td colspan="5" class="muted">No sentences match.</td></tr>';
  }

  /* --------------------------------------------------------- attribution -- */
  function renderAttribution(result) {
    const attr = result.attribution;
    if (!attr) return;
    window.Charts.bars($('#familyBars'), attr.families, 8);
    window.Charts.bars($('#opBars'), attr.operations, 9);
    $('#attrNote').textContent = attr.note;
  }

  /* ------------------------------------------------------------- windows -- */
  function renderWindows(result) {
    const threshold = result.decision.threshold;
    window.Charts.windowChart($('#windowChart'), result.chunks, threshold);

    const text = State.text || '';
    $('#windowTable tbody').innerHTML = result.chunks.map(function (c) {
      const excerpt = text.slice(c.char_start, c.char_start + 90).replace(/\s+/g, ' ');
      return '<tr>'
        + '<td class="col-num">' + (c.index + 1) + '</td>'
        + '<td class="col-num muted">' + c.token_start + '–' + c.token_end + '</td>'
        + '<td class="col-num" style="color:' + U.heatColor(c.score - threshold) + '">'
        + fmt.signed(c.score) + '</td>'
        + '<td class="col-num">' + fmt.pct(c.probability, 1) + '</td>'
        + '<td>' + (c.flagged
          ? '<span class="pill pill-ai">flagged</span>'
          : '<span class="pill pill-human">clear</span>') + '</td>'
        + '<td class="cell-text muted">' + escapeHtml(excerpt) + '…</td>'
        + '</tr>';
    }).join('');
  }

  /* ---------------------------------------------------------------- stats -- */
  function renderStats(result) {
    const s = result.statistics;
    $('#statGrid').innerHTML = [
      stat('Words', fmt.int(s.words)),
      stat('Sentences', fmt.int(s.sentences)),
      stat('Paragraphs', fmt.int(s.paragraphs)),
      stat('Tokens', fmt.int(result.tokens.count), 'model tokenizer'),
      stat('Unique words', fmt.int(s.unique_words), 'TTR ' + s.type_token_ratio),
      stat('Avg sentence', s.avg_sentence_words + ' w', 'sd ' + s.sentence_length_sd),
      stat('Avg word', s.avg_word_length + ' ch'),
      stat('Scored', fmt.duration(result.timing.seconds), result.timing.device.toUpperCase()),
      stat('Pooled tokens', fmt.int(result.tokens.pooled_tokens),
        'top ' + Math.round(result.tokens.rho * 100) + '%')
    ].join('');

    const all = result.thresholds.all || {};
    const strata = (State.config && State.config.strata) || [];
    const current = result.decision.stratum;
    $('#calTable tbody').innerHTML = strata
      .filter((row) => all[row.key])
      .map(function (row) {
        const t = all[row.key];
        return '<tr class="' + (row.key === current ? 'row-current' : '') + '">'
          + '<td>' + escapeHtml(row.label) + '</td>'
          + '<td class="col-num">' + fmt.signed(t['fpr_0.01']) + '</td>'
          + '<td class="col-num">' + fmt.signed(t['fpr_0.05']) + '</td>'
          + '<td class="col-num">' + fmt.signed(t['fpr_0.1']) + '</td>'
          + '<td class="col-num muted">' + calibrationN(row.key) + '</td>'
          + '</tr>';
      }).join('');
  }

  function calibrationN(key) {
    const cal = State.model && State.model.calibration;
    if (!cal) return '—';
    const row = cal.strata.find((s) => s.key === key);
    return row ? fmt.int(row.n) : '—';
  }

  function stat(label, value, sub) {
    return '<div class="stat"><div class="stat-label">' + escapeHtml(label) + '</div>'
      + '<div class="stat-value">' + escapeHtml(value) + '</div>'
      + (sub ? '<div class="stat-sub">' + escapeHtml(sub) + '</div>' : '') + '</div>';
  }

  /* =================================================================== batch */
  function bindBatch() {
    const drop = $('#batchDrop');
    const input = $('#batchInput');

    drop.addEventListener('click', () => input.click());
    input.addEventListener('change', function (e) {
      addBatchFiles(Array.from(e.target.files));
      e.target.value = '';
    });
    ['dragenter', 'dragover'].forEach((evt) =>
      drop.addEventListener(evt, (e) => { e.preventDefault(); drop.classList.add('is-dragging'); }));
    ['dragleave', 'drop'].forEach((evt) =>
      drop.addEventListener(evt, (e) => { e.preventDefault(); drop.classList.remove('is-dragging'); }));
    drop.addEventListener('drop', (e) => addBatchFiles(Array.from(e.dataTransfer.files)));

    $('#btnBatchRun').addEventListener('click', runBatch);
    $('#btnBatchCancel').addEventListener('click', async function () {
      if (!State.batchJob) return;
      try { await window.API.cancelBatch(State.batchJob.id); U.toast('Cancelling…', 'info'); }
      catch (err) { U.toast(err.message, 'error'); }
    });
    $('#btnBatchExport').addEventListener('click', function () {
      if (State.batchJob) window.location = '/api/batch/' + State.batchJob.id + '/export.csv';
    });
  }

  function addBatchFiles(files) {
    const seen = new Set(State.batchFiles.map((f) => f.name + f.size));
    files.forEach(function (f) {
      if (!seen.has(f.name + f.size)) { State.batchFiles.push(f); seen.add(f.name + f.size); }
    });
    renderBatchQueue();
  }

  function renderBatchQueue() {
    const host = $('#batchQueue');
    host.innerHTML = '';
    State.batchFiles.forEach(function (file, index) {
      const chip = el('div', { class: 'queue-chip' }, [
        el('span', { text: file.name }),
        el('span', { class: 'qc-size', text: fmt.bytes(file.size) }),
        el('button', {
          text: '×', title: 'Remove',
          onclick: function () { State.batchFiles.splice(index, 1); renderBatchQueue(); }
        })
      ]);
      host.appendChild(chip);
    });
    $('#btnBatchRun').disabled = State.batchFiles.length === 0;
  }

  async function runBatch() {
    if (!State.model || !State.model.loaded) {
      U.toast('Load the model before running a batch.', 'error');
      return;
    }
    try {
      const job = await window.API.startBatchFiles(State.batchFiles, {
        stratum: $('#batchStratum').value,
        fpr: $('#batchFpr').value
      });
      State.batchJob = job;
      $('#batchProgress').hidden = false;
      $('#btnBatchCancel').disabled = false;
      $('#btnBatchExport').disabled = false;
      if (job.skipped && job.skipped.length) {
        U.toast(job.skipped.length + ' file(s) could not be read.', 'error');
      }
      pollBatch();
    } catch (err) {
      U.toast(err.message, 'error');
    }
  }

  function pollBatch() {
    clearTimeout(State.batchTimer);
    if (!State.batchJob) return;

    window.API.batchStatus(State.batchJob.id).then(function (job) {
      State.batchJob = job;
      renderBatch(job);
      if (job.status === 'running' || job.status === 'queued') {
        State.batchTimer = setTimeout(pollBatch, 700);
      } else {
        $('#btnBatchCancel').disabled = true;
        U.toast('Batch ' + job.status + ' — ' + job.completed + '/' + job.total
          + ' documents, ' + job.flagged + ' flagged.', 'success');
      }
    }).catch((err) => U.toast(err.message, 'error'));
  }

  function renderBatch(job) {
    $('#batchBar').style.width = job.percent + '%';
    $('#batchStatus').textContent = job.status === 'running'
      ? 'Scoring…' : job.status.charAt(0).toUpperCase() + job.status.slice(1);
    $('#batchCount').textContent = job.completed + ' / ' + job.total
      + ' · ' + fmt.duration(job.elapsed);

    const done = job.items.filter((i) => i.status === 'done');
    const summary = $('#batchSummary');
    summary.hidden = done.length === 0;
    if (done.length) {
      const avg = done.reduce((a, i) => a + i.probability, 0) / done.length;
      summary.innerHTML = [
        stat('Documents', fmt.int(job.total)),
        stat('Flagged', fmt.int(job.flagged), fmt.pct(job.flagged / Math.max(1, done.length), 0) + ' of scored'),
        stat('Mean P(AI)', fmt.pct(avg, 1)),
        stat('Errors', fmt.int(job.items.filter((i) => i.status === 'error').length))
      ].join('');
    }

    $('#batchTable tbody').innerHTML = job.items.map(function (item) {
      if (item.status !== 'done') {
        return '<tr><td>' + escapeHtml(item.name) + '</td>'
          + '<td colspan="5" class="muted">' + escapeHtml(item.error || '—') + '</td>'
          + '<td><span class="pill pill-neutral">' + item.status + '</span></td></tr>';
      }
      return '<tr>'
        + '<td>' + escapeHtml(item.name) + '</td>'
        + '<td class="col-num muted">' + fmt.int(item.words) + '</td>'
        + '<td class="col-num">' + fmt.pct(item.probability, 1) + '</td>'
        + '<td class="col-num">' + fmt.signed(item.score) + '</td>'
        + '<td><span class="pill pill-' + item.verdict + '">' + escapeHtml(item.verdict_label) + '</span></td>'
        + '<td class="muted">' + escapeHtml(item.top_family || '—') + '</td>'
        + '<td><span class="pill pill-neutral">done</span></td>'
        + '</tr>';
    }).join('');
  }

  /* ================================================================ compare */
  function bindCompare() {
    $('#btnCompare').addEventListener('click', async function () {
      const docs = [
        { name: $('#cmpNameA').value || 'Document A', text: $('#cmpTextA').value },
        { name: $('#cmpNameB').value || 'Document B', text: $('#cmpTextB').value }
      ];
      if (!docs[0].text.trim() || !docs[1].text.trim()) {
        U.toast('Fill in both documents.', 'error');
        return;
      }
      U.busy(true, 'Comparing…');
      try {
        const data = await window.API.compare({
          documents: docs,
          stratum: $('#cmpStratum').value,
          fpr: $('#cmpFpr').value
        });
        renderCompare(data);
      } catch (err) {
        U.toast(err.message, 'error');
      } finally {
        U.busy(false);
      }
    });
  }

  function renderCompare(data) {
    const hosts = [$('#cmpResultA'), $('#cmpResultB')];
    const scores = [];

    data.documents.forEach(function (entry, i) {
      const host = hosts[i];
      if (!host) return;
      if (entry.error) { host.innerHTML = '<p class="hint">' + escapeHtml(entry.error) + '</p>'; return; }
      const d = entry.result.decision;
      scores.push(d);
      host.innerHTML =
        '<div class="verdict-badge" style="--verdict-color:' + U.verdictColor(d.verdict)
        + ';--verdict-soft:' + U.verdictSoft(d.verdict) + '">' + escapeHtml(d.verdict_label) + '</div>'
        + '<div class="metrics" style="margin-top:10px">'
        + metric('P(AI)', fmt.pct(d.probability, 1))
        + metric('Raw score', fmt.signed(d.score))
        + metric('Margin', fmt.signed(d.margin))
        + metric('Words', fmt.int(entry.result.statistics.words))
        + '</div>';
    });

    const delta = $('#cmpDelta');
    if (scores.length === 2) {
      const diff = scores[0].score - scores[1].score;
      delta.hidden = false;
      delta.innerHTML = '<strong>' + fmt.signed(diff) + '</strong> raw-score difference — '
        + escapeHtml(data.documents[0].name) + ' scores '
        + (diff > 0 ? 'more' : 'less') + ' machine-like than '
        + escapeHtml(data.documents[1].name) + '. '
        + '<span class="hint">Both scored against the same '
        + fmt.signed(scores[0].threshold) + ' cut-off.</span>';
    } else {
      delta.hidden = true;
    }
  }

  /* ================================================================ history */
  function bindHistory() {
    $('#histSearch').addEventListener('input', U.debounce(function (e) {
      State.history.q = e.target.value;
      State.history.offset = 0;
      loadHistory();
    }, 250));
    $('#histFlagged').addEventListener('change', function (e) {
      State.history.flagged = e.target.checked;
      State.history.offset = 0;
      loadHistory();
    });
    $('#btnHistExport').addEventListener('click', () => { window.location = '/api/history/export.csv'; });
    $('#btnHistClear').addEventListener('click', async function () {
      if (!confirm('Delete every stored analysis? This cannot be undone.')) return;
      try {
        const r = await window.API.historyClear();
        U.toast('Deleted ' + r.deleted + ' records.', 'success');
        loadHistory();
      } catch (err) { U.toast(err.message, 'error'); }
    });
  }

  async function loadHistory() {
    try {
      const [data, stats] = await Promise.all([
        window.API.history({
          limit: State.history.limit,
          offset: State.history.offset,
          q: State.history.q,
          flagged_only: State.history.flagged
        }),
        window.API.historyStats()
      ]);
      State.history.total = data.total;

      $('#histStats').innerHTML = [
        stat('Analyses', fmt.int(stats.total)),
        stat('Flagged', fmt.int(stats.flagged),
          stats.total ? fmt.pct(stats.flagged / stats.total, 0) + ' of all' : ''),
        stat('Mean P(AI)', fmt.pct(stats.avg_probability || 0, 1)),
        stat('Words scored', fmt.int(stats.words))
      ].join('');

      $('#histTable tbody').innerHTML = data.items.map(function (row) {
        return '<tr>'
          + '<td class="muted">' + escapeHtml(fmt.when(row.created_at)) + '</td>'
          + '<td class="cell-text">' + escapeHtml((row.label || row.preview).slice(0, 90)) + '</td>'
          + '<td class="col-num muted">' + fmt.int(row.words) + '</td>'
          + '<td class="col-num">' + fmt.pct(row.probability, 1) + '</td>'
          + '<td><span class="pill pill-' + row.verdict + '">' + escapeHtml(row.verdict) + '</span></td>'
          + '<td class="muted">' + escapeHtml(row.source) + '</td>'
          + '<td><button class="link-btn" data-open="' + row.id + '">Open</button> '
          + '<button class="link-btn" data-del="' + row.id + '">Delete</button></td>'
          + '</tr>';
      }).join('') || '<tr><td colspan="7" class="muted">Nothing stored yet.</td></tr>';

      $$('#histTable [data-open]').forEach((b) =>
        b.addEventListener('click', () => openHistory(b.dataset.open)));
      $$('#histTable [data-del]').forEach((b) =>
        b.addEventListener('click', async function () {
          await window.API.historyDelete(b.dataset.del);
          loadHistory();
        }));

      renderPager();
    } catch (err) {
      U.toast(err.message, 'error');
    }
  }

  function renderPager() {
    const h = State.history;
    const from = h.total ? h.offset + 1 : 0;
    const to = Math.min(h.offset + h.limit, h.total);
    $('#histPager').innerHTML =
      '<span>' + from + '–' + to + ' of ' + fmt.int(h.total) + '</span>'
      + '<span class="pager-btns">'
      + '<button class="btn btn-ghost btn-sm" ' + (h.offset === 0 ? 'disabled' : '') + ' data-page="prev">Previous</button>'
      + '<button class="btn btn-ghost btn-sm" ' + (to >= h.total ? 'disabled' : '') + ' data-page="next">Next</button>'
      + '</span>';
    $$('#histPager [data-page]').forEach((b) =>
      b.addEventListener('click', function () {
        State.history.offset += b.dataset.page === 'next' ? h.limit : -h.limit;
        State.history.offset = Math.max(0, State.history.offset);
        loadHistory();
      }));
  }

  async function openHistory(id) {
    U.busy(true, 'Loading record…');
    try {
      const record = await window.API.historyGet(id);
      State.result = record.payload;
      State.text = record.text || '';
      $('#editor').value = State.text;
      updateCounts();
      renderResult(record.payload);
      $$('.nav-item').forEach((b) => b.classList.toggle('is-active', b.dataset.view === 'analyze'));
      $$('.view').forEach((v) => v.classList.toggle('is-active', v.dataset.view === 'analyze'));
      if (!record.text) {
        U.toast('Record opened. The original text was not stored, so the heatmap is unavailable.', 'info');
      }
    } catch (err) {
      U.toast(err.message, 'error');
    } finally {
      U.busy(false);
    }
  }

  /* ================================================================== model */
  function bindModelPanel() {
    $('#btnLoadModel').addEventListener('click', async function () {
      try {
        await window.API.loadModel($('#selDevice').value || null);
        U.toast('Loading started. The first run downloads ~1.6 GB.', 'info');
        pollModel(true);
      } catch (err) { U.toast(err.message, 'error'); }
    });
    $('#btnUnloadModel').addEventListener('click', async function () {
      try {
        State.model = await window.API.unloadModel();
        refreshModelPanel();
        U.toast('Model unloaded, memory released.', 'success');
      } catch (err) { U.toast(err.message, 'error'); }
    });
  }

  function pollModel(fast) {
    clearTimeout(State.modelTimer);
    window.API.modelStatus().then(function (status) {
      const wasLoaded = State.model && State.model.loaded;
      State.model = status;
      renderModelPill(status);
      refreshModelPanel();
      if (!wasLoaded && status.loaded) {
        U.toast('Model ready on ' + status.device.toUpperCase()
          + ' (' + fmt.duration(status.load_seconds) + ')', 'success');
      }
      const busy = status.busy || fast;
      State.modelTimer = setTimeout(() => pollModel(status.busy), busy ? 800 : 6000);
    }).catch(function () {
      State.modelTimer = setTimeout(pollModel, 4000);
    });
  }

  function renderModelPill(status) {
    const pill = $('#modelPill');
    pill.classList.remove('is-ready', 'is-busy', 'is-error');

    if (status.error) {
      pill.classList.add('is-error');
      $('#modelPillTitle').textContent = 'Model error';
      $('#modelPillSub').textContent = status.error.slice(0, 60);
    } else if (status.busy) {
      pill.classList.add('is-busy');
      const p = status.progress;
      $('#modelPillTitle').textContent = p.state === 'downloading'
        ? 'Downloading ' + p.percent.toFixed(0) + '%' : 'Loading…';
      $('#modelPillSub').textContent = p.state === 'downloading'
        ? fmt.bytes(p.downloaded) + ' / ' + fmt.bytes(p.total) : p.message;
    } else if (status.loaded) {
      pill.classList.add('is-ready');
      $('#modelPillTitle').textContent = 'Model ready';
      $('#modelPillSub').textContent = status.device.toUpperCase()
        + (status.model ? ' · ' + status.model.version : '');
    } else {
      $('#modelPillTitle').textContent = status.downloaded ? 'Not loaded' : 'Not downloaded';
      $('#modelPillSub').textContent = status.downloaded
        ? 'Open Model to load' : '~1.6 GB download';
    }
  }

  function refreshModelPanel() {
    const status = State.model;
    if (!status) return;

    const progress = status.progress || {};
    const showProgress = status.busy || progress.state === 'downloading';
    $('#modelProgress').hidden = !showProgress;
    if (showProgress) {
      $('#modelBar').style.width = progress.percent + '%';
      $('#modelProgressText').textContent = progress.message || progress.state;
      $('#modelProgressPct').textContent = progress.total
        ? fmt.bytes(progress.downloaded) + ' / ' + fmt.bytes(progress.total)
        : progress.percent.toFixed(0) + '%';
    }

    $('#btnLoadModel').disabled = status.busy || status.loaded;
    $('#btnUnloadModel').disabled = !status.loaded;
    $('#btnLoadModel').textContent = status.loaded
      ? 'Loaded'
      : status.downloaded ? 'Load model' : 'Download & load (1.6 GB)';

    const m = status.model;
    $('#modelStats').innerHTML = [
      stat('Status', status.loaded ? 'Ready' : status.busy ? 'Working' : 'Idle',
        status.error ? 'error' : ''),
      stat('Device', (status.device || '—').toUpperCase(),
        status.cuda_available ? 'CUDA available' : 'CPU only'),
      stat('Parameters', m ? fmt.int(m.parameters) : '—'),
      stat('Context', m ? fmt.int(m.max_length) + ' tok' : '—'),
      stat('Load time', status.load_seconds ? fmt.duration(status.load_seconds) : '—'),
      stat('Version', m ? String(m.version) : '—')
    ].join('');

    const cal = status.calibration;
    $('#modelMeta').innerHTML = [
      '<div class="meta-block"><h4>Checkpoint</h4>'
      + metaRow('Repository', status.repo_id)
      + metaRow('Architecture', m ? m.architecture : '—')
      + metaRow('Style rank', m ? m.style_rank : '—')
      + metaRow('Pooling ρ', m ? m.rho : '—')
      + metaRow('Human anchors', m ? m.n_human_anchors : '—')
      + metaRow('Training step', m ? m.released_step : '—')
      + metaRow('Path', status.path || 'not downloaded')
      + '</div>',

      '<div class="meta-block"><h4>Families (' + (m ? m.families.length : 0) + ')</h4>'
      + '<div class="tag-list">'
      + (m ? m.families.map((f) => '<span class="tag">' + escapeHtml(f) + '</span>').join('') : '')
      + '</div>'
      + '<h4 style="margin-top:14px">Operations (' + (m ? m.ops.length : 0) + ')</h4>'
      + '<div class="tag-list">'
      + (m ? m.ops.map((o) => '<span class="tag">' + escapeHtml(o) + '</span>').join('') : '')
      + '</div></div>',

      '<div class="meta-block"><h4>Calibration</h4>'
      + (cal
        ? metaRow('Human validation texts', fmt.int(cal.n_human))
          + cal.strata.map((s) => metaRow(s.label, fmt.signed(s.thresholds['fpr_0.01']))).join('')
        : '<p class="hint">Load the model to read its calibration table.</p>')
      + '</div>'
    ].join('');
  }

  function metaRow(label, value) {
    return '<div class="meta-row"><span>' + escapeHtml(label)
      + '</span><span>' + escapeHtml(String(value)) + '</span></div>';
  }

  /* ================================================================== guide */
  function renderGuide() {
    $('#guide').innerHTML = `
      <h3>What MELD actually outputs</h3>
      <p>MELD is a 395M-parameter encoder that produces a <strong>raw score</strong> for a document.
      The probability shown on the gauge is just <code>sigmoid(raw score)</code> — it is a squashed
      view of the same number, <em>not</em> a calibrated "chance this is AI".</p>
      <p>The document score is the mean of the most machine-like 25% of tokens
      (<code>rho = 0.25</code>), which is why a mostly-human document containing a few generated
      paragraphs can still score high — and why the sentence heatmap is worth reading.</p>

      <h3>Why the cut-off is not 0.5</h3>
      <p>The checkpoint ships thresholds measured on 12,461 human documents: the score below which
      99%, 95% or 90% of genuine human writing fell. Picking <strong>1% FPR</strong> means roughly one
      in a hundred human documents of that type would be wrongly flagged. Those thresholds differ a
      lot by document type, so set <strong>Document type</strong> to match what you are checking.</p>

      <div class="callout">
        <strong>Never treat a flag as proof.</strong> At 1% FPR, scanning 500 human essays still
        produces about five false accusations. Use MELD to prioritise a human review, never as the
        sole basis for a decision about a person.
      </div>

      <h3>The 100-word floor</h3>
      <p>Short text scores high regardless of who wrote it. Below 100 words the result is noise, and
      the app says so. The length bar under the editor tracks this.</p>

      <h3>Documents longer than the context</h3>
      <p>MELD reads 2,048 tokens at a time. Longer documents are cut into overlapping windows; where
      windows overlap, each token keeps the score from the window in which it sat furthest from an
      edge. The document score then pools the top 25% of tokens across the whole text, and the
      <strong>Windows</strong> tab shows each window separately.</p>

      <h3>Attribution</h3>
      <p>The checkpoint carries 11 family prototypes and 9 edit-operation prototypes. The Attribution
      tab reads the softmax over those prototypes for the tokens that drove the score. This is
      <strong>indicative only</strong>: the model card publishes no accuracy figures for it, so treat
      it as a stylistic lean rather than an identification of a specific model.</p>

      <h3>Known weak spots</h3>
      <ul>
        <li>Non-native English writing scores higher on many detectors; MELD is calibrated on human
            text but that calibration set may not match your population.</li>
        <li>Heavily edited AI text, and AI-assisted human text, sit in the middle — the
            "Inconclusive" band exists for a reason.</li>
        <li>Technical and formulaic prose (documentation, boilerplate, legal text) is naturally
            low-variance and can read as machine-like.</li>
        <li>The model is English-only.</li>
      </ul>

      <h3>Everything here is scriptable</h3>
      <p>The UI is a client of this app's own REST API. Browse it at
      <a href="/docs" target="_blank" rel="noopener">/docs</a> and drive the same
      analysis from scripts.</p>`;
  }
})();
