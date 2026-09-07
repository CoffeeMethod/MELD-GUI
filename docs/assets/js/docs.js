/* MELD GUI docs — theme, mobile nav, scrollspy, copy buttons.
 * No dependencies; the site works without JS, this only adds convenience.
 */
(function () {
  'use strict';

  var $  = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ------------------------------------------------------------- theme --- */
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem('meld-docs-theme');
    if (saved) {
      root.setAttribute('data-theme', saved);
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      root.setAttribute('data-theme', 'light');
    }
  } catch (e) { /* storage blocked — keep the default */ }

  var toggle = $('#themeToggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('meld-docs-theme', next); } catch (e) { /* blocked */ }
    });
  }

  /* -------------------------------------------------------- mobile nav --- */
  var nav = $('#sidenav');
  var scrim = $('#scrim');
  var burger = $('#burger');

  function closeNav() {
    if (!nav) return;
    nav.classList.remove('is-open');
    if (scrim) scrim.classList.remove('is-open');
  }

  if (burger) {
    burger.addEventListener('click', function () {
      nav.classList.toggle('is-open');
      if (scrim) scrim.classList.toggle('is-open');
    });
  }
  if (scrim) scrim.addEventListener('click', closeNav);
  $$('.sidenav a').forEach(function (a) { a.addEventListener('click', closeNav); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeNav(); });

  /* --------------------------------------------------------- scrollspy --- */
  var links = $$('.sidenav a[href^="#"]');
  var byId = {};
  var sections = [];

  links.forEach(function (link) {
    var id = link.getAttribute('href').slice(1);
    var target = document.getElementById(id);
    if (!target) return;
    byId[id] = link;
    sections.push(target);
  });

  function highlight(id) {
    links.forEach(function (l) { l.classList.remove('is-current'); });
    if (byId[id]) byId[id].classList.add('is-current');
  }

  if ('IntersectionObserver' in window && sections.length) {
    // Track which headings are in the upper band of the viewport; the topmost
    // visible one wins, which keeps the highlight stable while scrolling up.
    var visible = new Set();
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) visible.add(entry.target.id);
        else visible.delete(entry.target.id);
      });
      for (var i = 0; i < sections.length; i++) {
        if (visible.has(sections[i].id)) { highlight(sections[i].id); return; }
      }
    }, { rootMargin: '-70px 0px -68% 0px', threshold: 0 });

    sections.forEach(function (s) { observer.observe(s); });
  }

  /* ------------------------------------------------------ copy buttons --- */
  $$('pre').forEach(function (pre) {
    var button = document.createElement('button');
    button.className = 'copy-btn';
    button.type = 'button';
    button.textContent = 'Copy';
    button.setAttribute('aria-label', 'Copy code to clipboard');

    button.addEventListener('click', function () {
      var code = pre.querySelector('code');
      var text = code ? code.textContent : pre.textContent;

      var done = function (ok) {
        button.textContent = ok ? 'Copied' : 'Failed';
        button.classList.toggle('done', ok);
        setTimeout(function () {
          button.textContent = 'Copy';
          button.classList.remove('done');
        }, 1600);
      };

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done(true); },
                                                 function () { done(false); });
      } else {
        // Fallback for pages not served over a secure context.
        var area = document.createElement('textarea');
        area.value = text;
        area.style.position = 'fixed';
        area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        var ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        area.remove();
        done(ok);
      }
    });

    pre.appendChild(button);
  });
})();
