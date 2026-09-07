/* MELD Detector — REST client.
 * Thin wrapper over fetch that turns FastAPI's `detail` field into a real
 * Error, so every caller can just try/catch.
 */
(function (global) {
  'use strict';

  async function request(path, options) {
    const opts = Object.assign({ headers: {} }, options || {});
    if (opts.body && !(opts.body instanceof FormData)) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(opts.body);
    }

    let response;
    try {
      response = await fetch(path, opts);
    } catch (err) {
      throw new Error('Cannot reach the server. Is it still running?');
    }

    if (!response.ok) {
      let detail = response.statusText || ('HTTP ' + response.status);
      try {
        const data = await response.json();
        if (data && data.detail) detail = data.detail;
      } catch (_) { /* non-JSON error body */ }
      const error = new Error(detail);
      error.status = response.status;
      throw error;
    }

    if (response.status === 204) return null;
    const type = response.headers.get('content-type') || '';
    return type.includes('application/json') ? response.json() : response.text();
  }

  const API = {
    request: request,

    // meta
    config:  () => request('/api/config'),
    health:  () => request('/api/health'),

    // model
    modelStatus: () => request('/api/model'),
    loadModel:   (device) => request('/api/model/load', { method: 'POST', body: { device: device || null } }),
    unloadModel: () => request('/api/model/unload', { method: 'POST' }),

    // analysis
    analyze: (payload) => request('/api/analyze', { method: 'POST', body: payload }),
    analyzeFile: (file, opts) => {
      const form = new FormData();
      form.append('file', file);
      form.append('stratum', opts.stratum);
      form.append('fpr', opts.fpr);
      form.append('save_history', String(opts.saveHistory !== false));
      return request('/api/analyze/file', { method: 'POST', body: form });
    },
    extract: (file) => {
      const form = new FormData();
      form.append('file', file);
      return request('/api/extract', { method: 'POST', body: form });
    },
    compare: (payload) => request('/api/compare', { method: 'POST', body: payload }),

    // batch
    startBatch: (payload) => request('/api/batch', { method: 'POST', body: payload }),
    startBatchFiles: (files, opts) => {
      const form = new FormData();
      files.forEach((f) => form.append('files', f));
      form.append('stratum', opts.stratum);
      form.append('fpr', opts.fpr);
      form.append('save_history', String(opts.saveHistory !== false));
      return request('/api/batch/files', { method: 'POST', body: form });
    },
    batchStatus: (id, withResults) =>
      request('/api/batch/' + id + (withResults ? '?results=true' : '')),
    cancelBatch: (id) => request('/api/batch/' + id + '/cancel', { method: 'POST' }),

    // history
    history: (params) => {
      const q = new URLSearchParams(params || {}).toString();
      return request('/api/history' + (q ? '?' + q : ''));
    },
    historyStats:  () => request('/api/history/stats'),
    historyGet:    (id) => request('/api/history/' + id),
    historyDelete: (id) => request('/api/history/' + id, { method: 'DELETE' }),
    historyClear:  () => request('/api/history', { method: 'DELETE' })
  };

  global.API = API;
})(window);
