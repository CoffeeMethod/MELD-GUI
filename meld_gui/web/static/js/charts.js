/* MELD Detector — SVG chart primitives.
 * Hand-rolled rather than pulled from a chart library: the shapes here are
 * few and specific, and this keeps the app dependency-free and offline.
 */
(function (global) {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  function svgEl(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    return node;
  }

  /* Point on the gauge arc. t=0 is the left end, t=1 the right. */
  function arcPoint(cx, cy, r, t) {
    const angle = Math.PI * (1 - t);
    return [cx + r * Math.cos(angle), cy - r * Math.sin(angle)];
  }

  function arcPath(cx, cy, r, from, to) {
    const [x0, y0] = arcPoint(cx, cy, r, from);
    const [x1, y1] = arcPoint(cx, cy, r, to);
    const large = to - from > 0.5 ? 1 : 0;
    return `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`;
  }

  /**
   * Semicircular P(AI) gauge with a tick showing where the calibrated
   * threshold sits, so the reading is always relative to the decision in use.
   */
  function gauge(svg, probability, threshold, color) {
    svg.innerHTML = '';
    const cx = 100, cy = 106, r = 76, width = 13;

    const track = svgEl('path', {
      d: arcPath(cx, cy, r, 0, 1),
      stroke: global.U.cssVar('--surface-3'),
      'stroke-width': width, 'stroke-linecap': 'round', fill: 'none'
    });
    svg.appendChild(track);

    const value = Math.max(0.001, Math.min(1, probability));
    const arc = svgEl('path', {
      d: arcPath(cx, cy, r, 0, value),
      stroke: color, 'stroke-width': width, 'stroke-linecap': 'round', fill: 'none'
    });
    svg.appendChild(arc);

    try {
      const total = arc.getTotalLength();
      arc.style.strokeDasharray = total;
      arc.style.strokeDashoffset = total;
      arc.style.transition = 'stroke-dashoffset .7s cubic-bezier(.4,0,.2,1)';
      requestAnimationFrame(() => { arc.style.strokeDashoffset = '0'; });
    } catch (_) { /* getTotalLength unsupported */ }

    // Threshold tick — P(AI) at the calibrated cut-off.
    const tAt = Math.max(0, Math.min(1, sigmoid(threshold)));
    const [ix, iy] = arcPoint(cx, cy, r - width / 2 - 2, tAt);
    const [ox, oy] = arcPoint(cx, cy, r + width / 2 + 2, tAt);
    svg.appendChild(svgEl('line', {
      x1: ix, y1: iy, x2: ox, y2: oy,
      stroke: global.U.cssVar('--text-dim'), 'stroke-width': 2, 'stroke-linecap': 'round'
    }));

    const [lx, ly] = arcPoint(cx, cy, r + 17, tAt);
    const label = svgEl('text', {
      x: lx, y: ly, 'text-anchor': 'middle', 'dominant-baseline': 'middle',
      fill: global.U.cssVar('--text-faint'), 'font-size': '8.5', 'font-weight': '600'
    });
    label.textContent = 'cut-off';
    label.setAttribute('stroke', 'none');
    svg.appendChild(label);

    ['0', '1'].forEach((text, i) => {
      const [x, y] = arcPoint(cx, cy, r + 15, i);
      const t = svgEl('text', {
        x: x, y: y + 4, 'text-anchor': 'middle',
        fill: global.U.cssVar('--text-faint'), 'font-size': '9'
      });
      t.textContent = text;
      t.setAttribute('stroke', 'none');
      svg.appendChild(t);
    });
  }

  /**
   * Linear raw-score scale. MELD's decision happens in raw-score space, so
   * this is the chart that actually explains a verdict: where the document
   * landed, and how far that is from the cut-off.
   */
  function scoreScale(host, score, threshold) {
    const lo = Math.min(-6, score - 2, threshold - 2);
    const hi = Math.max(8, score + 2, threshold + 2);
    const pos = (v) => ((v - lo) / (hi - lo)) * 100;

    const scoreAt = Math.max(0, Math.min(100, pos(score)));
    const threshAt = Math.max(0, Math.min(100, pos(threshold)));
    const margin = score - threshold;
    const color = global.U.heatColor(margin);

    host.innerHTML = `
      <div class="scale-head">
        <span>Raw score</span>
        <span class="scale-margin" style="color:${color}">
          ${global.U.fmt.signed(margin)} vs cut-off
        </span>
      </div>
      <div class="scale-track">
        <div class="scale-human" style="width:${threshAt}%"></div>
        <div class="scale-thresh" style="left:${threshAt}%"></div>
        <div class="scale-dot" style="left:${scoreAt}%;background:${color};
             box-shadow:0 0 0 4px ${global.U.heatColor(margin, 0.22)}"></div>
      </div>
      <div class="scale-foot">
        <span>${global.U.fmt.num(lo, 0)}</span>
        <span class="scale-legend">
          <b style="color:${color}">${global.U.fmt.signed(score)}</b> ·
          cut-off ${global.U.fmt.signed(threshold)}
        </span>
        <span>${global.U.fmt.num(hi, 0)}</span>
      </div>`;
  }

  /** Horizontal probability bars, used for family and operator attribution. */
  function bars(host, rows, limit) {
    host.innerHTML = '';
    const shown = rows.slice(0, limit || rows.length);
    const peak = Math.max.apply(null, shown.map((r) => r.probability).concat([0.0001]));

    shown.forEach((row, index) => {
      const wrap = global.U.el('div', { class: 'bar-row' + (index === 0 ? ' is-top' : '') });
      wrap.appendChild(global.U.el('div', { class: 'bar-name', text: row.label, title: row.label }));

      const track = global.U.el('div', { class: 'bar-track' });
      const fill = global.U.el('div', { class: 'bar-fill' });
      fill.style.width = '0%';
      track.appendChild(fill);
      wrap.appendChild(track);

      wrap.appendChild(global.U.el('div', {
        class: 'bar-value', text: global.U.fmt.pct(row.probability, 1)
      }));
      host.appendChild(wrap);

      requestAnimationFrame(() => {
        fill.style.width = (row.probability / peak * 100).toFixed(1) + '%';
      });
    });
  }

  /** Per-window score bars for documents longer than the model context. */
  function windowChart(host, chunks, threshold) {
    host.innerHTML = '';
    if (!chunks.length) return;

    const values = chunks.map((c) => c.score);
    const lo = Math.min.apply(null, values.concat([threshold])) - 1;
    const hi = Math.max.apply(null, values.concat([threshold])) + 1;
    const span = hi - lo || 1;

    chunks.forEach((chunk) => {
      const height = Math.max(4, ((chunk.score - lo) / span) * 100);
      const bar = global.U.el('div', {
        class: 'window-bar',
        title: `Window ${chunk.index + 1}\nscore ${global.U.fmt.signed(chunk.score)}\n`
             + `P(AI) ${global.U.fmt.pct(chunk.probability)}\n`
             + `tokens ${chunk.token_start}–${chunk.token_end}`
      });
      bar.style.height = '2%';
      bar.style.background = global.U.heatColor(chunk.score - threshold, 0.85);
      host.appendChild(bar);
      requestAnimationFrame(() => { bar.style.height = height + '%'; });
    });
  }

  global.Charts = {
    gauge: gauge,
    scoreScale: scoreScale,
    bars: bars,
    windowChart: windowChart,
    sigmoid: sigmoid
  };
})(window);
