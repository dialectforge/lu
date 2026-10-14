/* Lu portal service worker — app-shell cache.
 *
 * Two hard rules:
 *   1. /auth, /mcp and /health are NEVER cached and NEVER served from
 *      cache. A stale token or a stale answer is worse than an error.
 *   2. Code (index.html, app.js, app.css, the manifest) is NETWORK-FIRST.
 *      Cache-first on code means a shipped fix cannot reach an installed
 *      client until the cache name changes — the exact failure this file
 *      is most likely to cause. Only immutable media is cache-first.
 *
 * Bump CACHE on any change here so `activate` purges the old one.
 */

var CACHE = 'lu-shell-v2';

// Precached at install. Deliberately excludes lu_thinking_96.gif (~470 KB):
// it is fetched and cached on first use instead, so an install over
// cellular is small.
var SHELL = [
  '/',
  '/index.html',
  '/app.css',
  '/app.js',
  '/manifest.webmanifest',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/icon-maskable-512.png',
  '/icons/apple-touch-icon.png'
];

// Files whose content changes when we deploy. Network wins; cache is only
// the offline fallback.
var CODE = {
  '/': 1, '/index.html': 1, '/app.js': 1, '/app.css': 1, '/manifest.webmanifest': 1
};

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(CACHE).then(function (cache) {
      // addAll is all-or-nothing; add individually so one missing asset
      // cannot block installation of the whole shell.
      return Promise.all(SHELL.map(function (url) {
        return cache.add(new Request(url, { cache: 'reload' })).catch(function () {});
      }));
    }).then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (k) {
        return k === CACHE ? null : caches.delete(k);
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

function isApi(url) {
  return url.pathname === '/auth'
      || url.pathname === '/mcp'
      || url.pathname === '/health';
}

// Only store what we asked for. Without this an HTML error page served
// under an asset URL would be cached permanently under that URL.
function cacheable(req, res) {
  if (!res || res.status !== 200 || res.type !== 'basic') return false;
  var ct = res.headers.get('Content-Type') || '';
  if (req.destination === 'image' && ct.indexOf('image/') !== 0) return false;
  if (req.destination === 'script' && ct.indexOf('javascript') < 0) return false;
  if (req.destination === 'style' && ct.indexOf('text/css') < 0) return false;
  return true;
}

function networkFirst(req) {
  return fetch(req).then(function (res) {
    if (cacheable(req, res)) {
      var copy = res.clone();
      caches.open(CACHE).then(function (c) { c.put(req, copy); });
    }
    return res;
  }).catch(function () {
    return caches.match(req).then(function (hit) {
      return hit || caches.match('/index.html') || caches.match('/');
    });
  });
}

self.addEventListener('fetch', function (event) {
  var req = event.request;
  if (req.method !== 'GET') return;                 // /auth + /mcp are POST

  var url = new URL(req.url);
  if (url.origin !== self.location.origin) return;  // never touch cross-origin
  if (isApi(url)) return;                           // straight to the network

  // Navigations and code: network-first, cache only as an offline fallback.
  if (req.mode === 'navigate' || CODE[url.pathname]) {
    event.respondWith(networkFirst(req));
    return;
  }

  // Media (icons, the thinking GIF): cache-first, refreshed in background.
  event.respondWith(
    caches.match(req).then(function (hit) {
      var live = fetch(req).then(function (res) {
        if (cacheable(req, res)) {
          var copy = res.clone();
          caches.open(CACHE).then(function (c) { c.put(req, copy); });
        }
        return res;
      }).catch(function () { return hit; });
      return hit || live;
    })
  );
});
