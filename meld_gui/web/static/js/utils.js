/* MELD Detector — DOM helpers, formatting and the score→colour mapping. */
(function (global) {
  'use strict';

  const $  = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      if (value == null || value === false) return;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'html') node.innerHTML = value;
      else if (key.startsWith('on') && typeof value === 'function') {
        node.addEventListener(key.slice(2).toLowerCase(), value);
      } else node.setAttribute(key, value);
    });
    (Array.isArray(children) ? children : children ? [children] : [])
      .forEach((c) => node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
    return node;
  }

  const escapeHtml = (str) => String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  // ------------------------------------------------------------- formatting
  const fmt = {
    pct:   (v, d) => (v * 100).toFixed(d == null ? 1 : d) + '%',
    num:   (v, d) => Number(v).toFixed(d == null ? 2 : d),
    signed:(v, d) => (v >= 0 ? '+' : '') + Number(v).toFixed(d == null ? 2 : d),
    int:   (v) => Number(v).toLocaleString(),

    bytes: (v) => {
      if (!v) return '0 B';
      const units = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.min(units.length - 1, Math.floor(Math.log(v) / Math.log(1024)));
      return (v / Math.pow(1024, i)).toFixed(i ? 1 : 0) + ' ' + units[i];
    },

    duration: (s) => {
      if (s < 1) return (s * 1000).toFixed(0) + ' ms';
      if (s < 60) return s.toFixed(1) + ' s';
      const m = Math.floor(s / 60);
      return m + 'm ' + Math.round(s % 60) + 's';
    },

    when: (epochSeconds) => {
      const then = new Date(epochSeconds * 1000);
      const diff = (Date.now() - then.getTime()) / 1000;
      if (diff < 60) return 'just now';
      if (diff < 3600) return Math.floor(diff / 60) + ' min ago';
      if (diff < 86400) return Math.floor(diff / 3600) + ' h ago';
      if (diff < 604800) return Math.floor(diff / 86400) + ' d ago';
      return then.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    }
  };

  // ----------------------------------------------------------------- colour
  /* Verdict names come from the backend; keep this map in step with
     thresholds.VERDICT_LABELS. */
  const VERDICT_VARS = {
    human:     '--human',
    uncertain: '--uncertain',
    likely_ai: '--likely',
    ai:        '--ai'
  };

  const cssVar = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const verdictColor = (verdict) => cssVar(VERDICT_VARS[verdict] || '--text-faint');
  const verdictSoft  = (verdict) => cssVar((VERDICT_VARS[verdict] || '--text-faint') + '-soft')
    || 'transparent';

  /* Map a raw margin (score − threshold) onto the heat ramp.
     Negative margins read as human, positive as machine; the ramp is centred
     on the threshold rather than on zero, so the colours track the decision
     the user actually configured. */
  function heatColor(margin, alpha) {
    const t = Math.max(-1, Math.min(1, margin / 3));   // saturate beyond ±3
    const stops = [
      { at: -1.0, c: [34, 192, 138] },   // human
      { at: -0.2, c: [90, 170, 190] },
      { at:  0.0, c: [240, 180, 41] },   // threshold
      { at:  0.4, c: [251, 139, 60] },
      { at:  1.0, c: [244, 82, 95] }     // strong machine signal
    ];
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
      if (t >= stops[i].at && t <= stops[i + 1].at) { lo = stops[i]; hi = stops[i + 1]; break; }
    }
    const span = hi.at - lo.at || 1;
    const k = (t - lo.at) / span;
    const rgb = lo.c.map((v, i) => Math.round(v + (hi.c[i] - v) * k));
    return 'rgba(' + rgb.join(',') + ',' + (alpha == null ? 1 : alpha) + ')';
  }

  // ------------------------------------------------------------------ misc
  function debounce(fn, wait) {
    let timer;
    return function () {
      const args = arguments, ctx = this;
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(ctx, args), wait || 200);
    };
  }

  function download(filename, content, mime) {
    const blob = content instanceof Blob
      ? content : new Blob([content], { type: mime || 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = el('a', { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function toCsv(rows, columns) {
    const escape = (v) => {
      const s = v == null ? '' : String(v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    return [columns.join(',')]
      .concat(rows.map((r) => columns.map((c) => escape(r[c])).join(',')))
      .join('\n');
  }

  /* Lightweight JSON syntax highlighting for the raw-output tab. */
  function highlightJson(value) {
    const json = escapeHtml(JSON.stringify(value, null, 2));
    return json.replace(
      /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
      (match) => {
        let cls = 'n';
        if (/^"/.test(match)) cls = /:$/.test(match) ? 'k' : 's';
        else if (/true|false|null/.test(match)) cls = 'b';
        return '<span class="' + cls + '">' + match + '</span>';
      }
    );
  }

  // --------------------------------------------------------------- toasts
  function toast(message, kind, timeout) {
    const host = $('#toasts');
    if (!host) return;
    const node = el('div', { class: 'toast toast-' + (kind || 'info'), text: message });
    host.appendChild(node);
    setTimeout(() => {
      node.classList.add('is-out');
      setTimeout(() => node.remove(), 220);
    }, timeout || (kind === 'error' ? 6500 : 3200));
  }

  function busy(on, text) {
    const overlay = $('#busyOverlay');
    if (!overlay) return;
    if (text) $('#busyText').textContent = text;
    overlay.hidden = !on;
  }

  global.U = {
    $: $, $$: $$, el: el, escapeHtml: escapeHtml,
    fmt: fmt, cssVar: cssVar,
    verdictColor: verdictColor, verdictSoft: verdictSoft, heatColor: heatColor,
    debounce: debounce, download: download, toCsv: toCsv,
    highlightJson: highlightJson, toast: toast, busy: busy
  };
})(window);
