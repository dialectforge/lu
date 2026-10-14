/* Lu portal — vanilla JS, no build step, no dependencies, no npm.
 *
 * Talks to the lu_guest_http guest surface (crates/lu_guest_http):
 *   POST /auth  {"passphrase":"…"} | {"invite_code":"…"}
 *               -> 200 {"token":"…","principal":"owner"|"guest:<code>"}
 *               -> 400 provide 'passphrase' or 'invite_code'
 *               -> 401 invalid credentials   (bad passphrase, unknown code,
 *                                             AND already-redeemed code —
 *                                             the server deliberately does
 *                                             not distinguish them, and
 *                                             neither do we)
 *   POST /mcp   Authorization: Bearer <token>
 *               {"method":"tools/call",
 *                "params":{"name":"lu4d_ask","arguments":{"query":"…"}}}
 *               -> 200 {"result":{content:[{text}],structuredContent:{…}}}
 *               -> 200 {"error":{"code":…,"message":"…"}}
 *               -> 401 missing or invalid bearer token
 *
 * Server auth state is IN MEMORY. A restart of lu_guest_http invalidates
 * every issued token, so any 401 on /mcp means "sign in again" — it must
 * never leave the UI hanging.
 */
(function () {
  'use strict';

  /* --------------------------------------------------------------- */
  /* constants                                                        */
  /* --------------------------------------------------------------- */

  var TOKEN_KEY  = 'lu.token';
  var THREAD_KEY = 'lu.thread';

  // Why localStorage and not sessionStorage: invite codes are SINGLE USE.
  // A guest who closed the app with a session-scoped token would be locked
  // out permanently, because their code is already spent server-side. So
  // the token persists — but with a client-side expiry, so an unattended
  // phone is not an indefinite open door. Server-side revocation does not
  // exist on this surface; the token dies for real when the binary
  // restarts.
  var TOKEN_TTL_MS = 12 * 60 * 60 * 1000;

  // Caddy caps a query at 90s (deliberately under Cloudflare's 100s).
  // We do NOT abort earlier than the edge does — a slow answer is a real
  // answer. This ceiling only exists so a wedged socket cannot strand the
  // UI forever.
  var ASK_CEILING_MS = 150000;

  var COPY = {
    badCreds: 'That didn\u2019t work. Check the passphrase or invite code and try again.',
    noCreds:  'Enter a passphrase or an invite code first.',
    offline:  'Lu is offline right now \u2014 it goes quiet while it\u2019s training. Try again shortly.',
    unreach:  'Can\u2019t reach Lu. Check your connection and try again.',
    expired:  'Your session ended \u2014 Lu restarted. Sign in again.',
    slow:     'Lu took too long to answer. Ask again, or try a narrower question.',
    trouble:  'Something went wrong on Lu\u2019s side. Nothing was lost \u2014 try asking again.'
  };

  /* --------------------------------------------------------------- */
  /* dom                                                              */
  /* --------------------------------------------------------------- */

  function $(id) { return document.getElementById(id); }

  var viewLogin  = $('view-login');
  var viewChat   = $('view-chat');
  var loginForm  = $('login-form');
  var credInput  = $('cred');
  var credLabel  = $('cred-label');
  var loginBtn   = $('login-submit');
  var loginMsg   = $('login-msg');
  var loginFine  = $('login-fine');
  var tabPass    = $('tab-pass');
  var tabInvite  = $('tab-invite');
  var thread     = $('thread');
  var emptyState = $('empty');
  var composer   = $('composer');
  var codeInput  = $('code-input');
  var qInput     = $('q');
  var sendBtn    = $('send');
  var logoutBtn  = $('logout');
  var whoLabel   = $('who');
  var banner     = $('banner');

  /* --------------------------------------------------------------- */
  /* state                                                            */
  /* --------------------------------------------------------------- */

  var mode = 'passphrase';   // 'passphrase' | 'invite'
  var token = null;
  var principal = null;
  var busy = false;
  var turns = [];            // [{who:'you'|'lu', kind, text, sources}]

  // Every request captures the epoch it was issued under. Signing out,
  // signing in, or being bounced by a 401 bumps the epoch, so a reply that
  // arrives after the session changed is dropped instead of leaking the
  // previous principal's answer into the new session or destroying the
  // freshly issued token.
  var epoch = 0;
  var inflight = null;

  function endSession() {
    epoch++;
    if (inflight && inflight.abort) { try { inflight.abort(); } catch (e) {} }
    inflight = null;
    setBusy(false);
  }

  function store(key, val) {
    try { val === null ? localStorage.removeItem(key) : localStorage.setItem(key, val); }
    catch (e) { /* private mode / storage disabled — run in memory */ }
  }
  function load(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }

  function saveToken(t, who) {
    store(TOKEN_KEY, JSON.stringify({ t: t, who: who, exp: Date.now() + TOKEN_TTL_MS }));
  }
  function readToken() {
    var raw = load(TOKEN_KEY);
    if (!raw) return null;
    try {
      var rec = JSON.parse(raw);
      if (!rec || !rec.t || !rec.exp || Date.now() > rec.exp) return null;
      return rec;
    } catch (e) { return null; }
  }
  function dropToken() { store(TOKEN_KEY, null); }

  function saveThread() {
    try { sessionStorage.setItem(THREAD_KEY, JSON.stringify(turns.slice(-40))); }
    catch (e) {}
  }
  function loadThread() {
    try {
      var raw = sessionStorage.getItem(THREAD_KEY);
      var parsed = raw ? JSON.parse(raw) : null;
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) { return []; }
  }
  function clearThread() {
    turns = [];
    try { sessionStorage.removeItem(THREAD_KEY); } catch (e) {}
  }

  /* --------------------------------------------------------------- */
  /* views                                                            */
  /* --------------------------------------------------------------- */

  function showLogin(message) {
    viewChat.hidden = true;
    viewChat.classList.remove('is-on');
    viewLogin.hidden = false;
    viewLogin.classList.add('is-on');
    loginMsg.textContent = message || '';
    credInput.value = '';
    setBusy(false);
    setBusyLogin(false);
  }

  function showChat() {
    viewLogin.hidden = true;
    viewLogin.classList.remove('is-on');
    viewChat.hidden = false;
    viewChat.classList.add('is-on');
    // The invite code inside "guest:<code>" is never displayed or stored.
    whoLabel.textContent = principal === 'owner'
      ? 'signed in as owner'
      : 'signed in as guest';
    renderThread();
  }

  function setBanner(text) {
    if (text) { banner.textContent = text; banner.hidden = false; }
    else { banner.textContent = ''; banner.hidden = true; }
  }

  function expire() {
    endSession();
    token = null;
    principal = null;
    dropToken();
    clearThread();
    showLogin(COPY.expired);
  }

  /* --------------------------------------------------------------- */
  /* transport                                                        */
  /* --------------------------------------------------------------- */

  // Returns {status, json, text, networkError, timedOut}. Never throws.
  // `handle`, if given, receives an .abort() so the caller can cancel.
  function post(path, body, headers, timeoutMs, handle) {
    var ctrl = ('AbortController' in window) ? new AbortController() : null;
    var timer = null;
    if (handle) { handle.abort = function () { if (ctrl) ctrl.abort(); }; }
    if (ctrl && timeoutMs) {
      timer = setTimeout(function () { ctrl.abort(); }, timeoutMs);
    }
    var init = {
      method: 'POST',
      headers: Object.assign({ 'Content-Type': 'application/json' }, headers || {}),
      body: JSON.stringify(body),
      cache: 'no-store'
    };
    if (ctrl) { init.signal = ctrl.signal; }

    return fetch(path, init).then(function (res) {
      if (timer) clearTimeout(timer);
      return res.text().then(function (text) {
        var json = null;
        try { json = JSON.parse(text); } catch (e) { /* Caddy error pages are plain text */ }
        return { status: res.status, json: json, text: text };
      });
    }).catch(function (err) {
      if (timer) clearTimeout(timer);
      var aborted = err && (err.name === 'AbortError');
      return { status: 0, json: null, text: '', networkError: true, timedOut: aborted };
    });
  }

  /* --------------------------------------------------------------- */
  /* login                                                            */
  /* --------------------------------------------------------------- */

  function setMode(next) {
    mode = next;
    var isPass = (next === 'passphrase');
    tabPass.classList.toggle('is-on', isPass);
    tabInvite.classList.toggle('is-on', !isPass);
    tabPass.setAttribute('aria-pressed', String(isPass));
    tabInvite.setAttribute('aria-pressed', String(!isPass));
    credLabel.textContent = isPass ? 'Owner passphrase' : 'Invite code';
    credInput.type = isPass ? 'password' : 'text';
    credInput.name = isPass ? 'password' : 'code';
    credInput.setAttribute('autocomplete', isPass ? 'current-password' : 'one-time-code');
    loginFine.textContent = isPass
      ? 'Only John has this.'
      : 'An invite code works once. Ask John for one.';
    loginMsg.textContent = '';
    credInput.value = '';
    credInput.focus();
  }

  function setBusyLogin(on) {
    loginBtn.disabled = on;
    credInput.disabled = on;
    loginBtn.textContent = on ? 'Signing in\u2026' : 'Sign in';
  }

  function doLogin(event) {
    event.preventDefault();
    if (loginBtn.disabled) return;
    var value = credInput.value.trim();
    if (!value) { loginMsg.textContent = COPY.noCreds; return; }

    var body = (mode === 'passphrase') ? { passphrase: value } : { invite_code: value };
    loginMsg.textContent = '';
    setBusyLogin(true);

    var mine = ++epoch;
    var handle = {};
    inflight = handle;

    post('/auth', body, null, 30000, handle).then(function (r) {
      if (mine !== epoch) return;
      inflight = null;
      setBusyLogin(false);

      if (r.networkError) {
        loginMsg.textContent = r.timedOut ? COPY.slow : COPY.unreach;
        return;
      }
      if (r.status === 200 && r.json && r.json.token) {
        token = r.json.token;
        principal = (r.json.principal === 'owner') ? 'owner' : 'guest';
        saveToken(token, principal);
        clearThread();
        setBanner(null);
        showChat();
        setTimeout(function () { qInput.focus(); }, 60);
        return;
      }
      // 401 covers bad passphrase, unknown code AND already-redeemed code.
      // One message for all three: never reveal whether a code exists.
      if (r.status === 401) { loginMsg.textContent = COPY.badCreds; return; }
      if (r.status === 400) { loginMsg.textContent = COPY.noCreds; return; }
      if (r.status === 502 || r.status === 503 || r.status === 504) {
        loginMsg.textContent = COPY.offline; return;
      }
      loginMsg.textContent = COPY.trouble;
    });
  }

  function doLogout() {
    endSession();
    token = null;
    principal = null;
    dropToken();
    clearThread();
    setBanner(null);
    showLogin('Signed out.');
  }

  /* --------------------------------------------------------------- */
  /* rendering                                                        */
  /* --------------------------------------------------------------- */

  function paragraphs(into, text) {
    var blocks = String(text == null ? '' : text).split(/\n\s*\n/);
    for (var i = 0; i < blocks.length; i++) {
      var block = blocks[i].trim();
      if (!block) continue;
      var p = document.createElement('p');
      var lines = block.split('\n');
      for (var j = 0; j < lines.length; j++) {
        if (j) p.appendChild(document.createElement('br'));
        // textContent only — Lu's output is never treated as markup.
        p.appendChild(document.createTextNode(lines[j]));
      }
      into.appendChild(p);
    }
    if (!into.childNodes.length) {
      into.appendChild(document.createTextNode(String(text || '')));
    }
  }

  function turnNode(turn) {
    var wrap = document.createElement('div');
    wrap.className = 'turn turn--' + (turn.who === 'you' ? 'you' : 'lu');
    if (turn.kind === 'refusal') wrap.classList.add('turn--refusal');
    if (turn.kind === 'trouble') wrap.classList.add('turn--trouble');

    var who = document.createElement('p');
    who.className = 'turn__who';
    who.textContent = (turn.who === 'you') ? 'You' : 'Lu';
    wrap.appendChild(who);

    var bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (turn.kind === 'code') {
      var intro = document.createElement('p');
      intro.textContent = turn.ok
        ? 'Here\u2019s a working program:'
        : 'I wrote a program but it didn\u2019t run cleanly, so I won\u2019t claim it works:';
      bubble.appendChild(intro);
      var pre = document.createElement('pre'); pre.className = 'code';
      var codeEl = document.createElement('code'); codeEl.textContent = turn.code;
      pre.appendChild(codeEl); bubble.appendChild(pre);
      if (turn.stdout) {
        var ol = document.createElement('p'); ol.className = 'code__out-label'; ol.textContent = 'Output';
        bubble.appendChild(ol);
        var out = document.createElement('pre'); out.className = 'code code--out'; out.textContent = turn.stdout;
        bubble.appendChild(out);
      }
      var dl = document.createElement('button'); dl.type = 'button';
      dl.className = 'btn btn--ghost code__dl'; dl.textContent = 'Download ' + turn.download;
      dl.addEventListener('click', function () {
        var blob = new Blob([turn.code], { type: 'text/x-python' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a'); a.href = url; a.download = turn.download;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
      });
      bubble.appendChild(dl);
      wrap.appendChild(bubble);
      return wrap;
    }

    if (turn.kind === 'refusal') {
      var tag = document.createElement('span');
      tag.className = 'tag';
      // Refusal is presented as Lu CHOOSING not to answer. It is the
      // anti-hallucination floor, not a failure.
      tag.textContent = 'Lu declined';
      bubble.appendChild(tag);
    }

    paragraphs(bubble, turn.text);

    if (turn.sources) {
      var s = document.createElement('p');
      s.className = 'sources';
      s.textContent = turn.sources;
      bubble.appendChild(s);
    }

    wrap.appendChild(bubble);
    return wrap;
  }

  function renderThread() {
    while (thread.firstChild) thread.removeChild(thread.firstChild);
    if (!turns.length) {
      thread.appendChild(emptyState);
      emptyState.hidden = false;
      return;
    }
    emptyState.hidden = true;
    for (var i = 0; i < turns.length; i++) {
      thread.appendChild(turnNode(turns[i]));
    }
    scrollDown();
  }

  function pushTurn(turn) {
    turns.push(turn);
    if (emptyState.parentNode === thread) thread.removeChild(emptyState);
    thread.appendChild(turnNode(turn));
    saveThread();
    scrollDown();
  }

  function scrollDown() {
    requestAnimationFrame(function () { thread.scrollTop = thread.scrollHeight; });
  }

  // Latency-aware working state: Lu's queries are slow by design, so the
  // UI counts up rather than pretending it is nearly done.
  function makeThinking() {
    var wrap = document.createElement('div');
    wrap.className = 'turn turn--lu';

    var who = document.createElement('p');
    who.className = 'turn__who';
    who.textContent = 'Lu';
    wrap.appendChild(who);

    var bubble = document.createElement('div');
    bubble.className = 'bubble thinking';

    var gif = document.createElement('img');
    gif.className = 'thinking__gif';
    gif.src = '/lu_thinking_96.gif';
    gif.alt = '';
    gif.width = 40; gif.height = 40;
    bubble.appendChild(gif);

    // Static stand-in shown instead of the GIF under prefers-reduced-motion.
    var dot = document.createElement('span');
    dot.className = 'thinking__dot';
    bubble.appendChild(dot);

    var txt = document.createElement('span');
    txt.className = 'thinking__txt';
    txt.textContent = 'Thinking';
    bubble.appendChild(txt);

    var clock = document.createElement('span');
    clock.className = 'thinking__clock';
    // The thread is an aria-live log; a per-second counter inside it would
    // be read aloud continuously for the whole wait.
    clock.setAttribute('aria-hidden', 'true');
    bubble.appendChild(clock);

    wrap.appendChild(bubble);

    var started = Date.now();
    var tick = setInterval(function () {
      var s = Math.round((Date.now() - started) / 1000);
      clock.textContent = s >= 3 ? (s + 's') : '';
      if (s === 20) txt.textContent = 'Still thinking \u2014 this one is deep';
      if (s === 60) txt.textContent = 'Still working. Long queries can take a minute or two';
    }, 1000);

    return {
      node: wrap,
      stop: function () {
        clearInterval(tick);
        if (wrap.parentNode) wrap.parentNode.removeChild(wrap);
      }
    };
  }

  /* --------------------------------------------------------------- */
  /* ask                                                              */
  /* --------------------------------------------------------------- */

  // A refusal is identifiable two ways, both first-class answers:
  //   structuredContent.shape  === 'Refusal'        (idk_gate)
  //   structuredContent.source === 'greet_refusal'  (pleasantry guard)
  function isRefusal(structured) {
    if (!structured || typeof structured !== 'object') return false;
    return structured.shape === 'Refusal' || structured.source === 'greet_refusal';
  }

  function answerText(result) {
    var sc = result && result.structuredContent;
    if (sc && typeof sc.answer === 'string' && sc.answer.trim()) return sc.answer;
    var content = result && result.content;
    if (Array.isArray(content)) {
      for (var i = 0; i < content.length; i++) {
        if (content[i] && typeof content[i].text === 'string') return content[i].text;
      }
    }
    return '';
  }

  function sourceLine(structured) {
    if (!structured) return '';
    if (typeof structured.source_summary === 'string' && structured.source_summary.trim()) {
      return structured.source_summary.trim();
    }
    var labels = structured.source_labels;
    if (Array.isArray(labels) && labels.length) {
      return 'Sources: ' + labels.join(', ');
    }
    return '';
  }

  function setBusy(on) {
    busy = on;
    sendBtn.disabled = on;
    qInput.readOnly = on;
  }

  function ask(query) {
    pushTurn({ who: 'you', kind: 'answer', text: query });
    setBusy(true);

    var pending = makeThinking();
    thread.appendChild(pending.node);
    scrollDown();

    var mine = epoch;
    var handle = {};
    inflight = handle;

    post('/mcp', {
      method: 'tools/call',
      params: { name: 'lu4d_ask', arguments: { query: query } }
    }, { 'Authorization': 'Bearer ' + token }, ASK_CEILING_MS, handle)
      .then(function (r) {
        pending.stop();
        // The session changed while this was in flight (sign-out, sign-in,
        // or an expiry). Drop the reply on the floor.
        if (mine !== epoch) return;
        inflight = null;
        setBusy(false);

        if (r.networkError) {
          pushTurn({ who: 'lu', kind: 'trouble',
                     text: r.timedOut ? COPY.slow : COPY.unreach });
          return;
        }

        // Server auth state is in-memory: a restart drops every token.
        // Return to login rather than hanging or silently failing.
        if (r.status === 401) { expire(); return; }

        if (r.status === 502 || r.status === 503 || r.status === 504) {
          pushTurn({ who: 'lu', kind: 'trouble', text: COPY.offline });
          return;
        }

        if (r.status !== 200 || !r.json) {
          pushTurn({ who: 'lu', kind: 'trouble', text: COPY.trouble });
          return;
        }

        // MCP-level error arrives inside a 200.
        if (r.json.error) {
          var m = r.json.error.message;
          pushTurn({ who: 'lu', kind: 'trouble',
                     text: COPY.trouble + (m ? '\n\n' + m : '') });
          return;
        }

        var result = r.json.result;
        var structured = result && result.structuredContent;
        var text = answerText(result);
        if (!text) { pushTurn({ who: 'lu', kind: 'trouble', text: COPY.trouble }); return; }

        var refusal = isRefusal(structured);
        pushTurn({
          who: 'lu',
          kind: refusal ? 'refusal' : 'answer',
          text: text,
          sources: refusal ? '' : sourceLine(structured)
        });
      });
  }

  function looksLikeCode(q) {
    var s = q.toLowerCase();
    if (/^(write|create|generate|implement|make me|give me|code)\b/.test(s)) return true;
    var kw = ['fizzbuzz','palindrome','fibonacci','factorial','prime','vowel',
      'duplicate line','longest line','word freq','word count','second largest',
      'a script','a program','a function','reverse a string','sort the numbers','average of'];
    for (var i = 0; i < kw.length; i++) if (s.indexOf(kw[i]) >= 0) return true;
    return false;
  }

  function askCode(prompt, input) {
    pushTurn({ who: 'you', kind: 'answer', text: prompt });
    setBusy(true);
    var pending = makeThinking();
    thread.appendChild(pending.node);
    scrollDown();
    var mine = epoch; var handle = {}; inflight = handle;
    var arg = { prompt: prompt };
    if (input && input.trim()) arg.input = input;
    post('/mcp', { method: 'tools/call', params: { name: 'lu4d_code', arguments: arg } },
         { 'Authorization': 'Bearer ' + token }, ASK_CEILING_MS, handle)
      .then(function (r) {
        pending.stop();
        if (mine !== epoch) return;
        inflight = null; setBusy(false);
        if (r.networkError) { pushTurn({ who: 'lu', kind: 'trouble', text: r.timedOut ? COPY.slow : COPY.unreach }); return; }
        if (r.status === 401) { expire(); return; }
        if (r.status !== 200 || !r.json) { pushTurn({ who: 'lu', kind: 'trouble', text: COPY.trouble }); return; }
        if (r.json.error) { pushTurn({ who: 'lu', kind: 'trouble', text: COPY.trouble + '\n\n' + (r.json.error.message || '') }); return; }
        var sc = (r.json.result && r.json.result.structuredContent) || {};
        if (!sc.code) {
          pushTurn({ who: 'lu', kind: 'refusal',
            text: answerText(r.json.result) || 'I can\u2019t ground this request, so I won\u2019t fake a solution.' });
          return;
        }
        pushTurn({ who: 'lu', kind: 'code', code: sc.code, stdout: sc.stdout || '',
          ok: !!sc.ok, download: sc.download_name || 'program.py' });
      });
  }

  function onSubmit(event) {
    event.preventDefault();
    if (busy) return;
    var q = qInput.value.trim();
    if (!q) return;
    qInput.value = '';
    autoGrow();
    if (looksLikeCode(q)) { askCode(q, codeInput.value); codeInput.value = ''; } else { ask(q); }
  }

  function autoGrow() {
    qInput.style.height = 'auto';
    qInput.style.height = Math.min(qInput.scrollHeight, window.innerHeight * 0.34) + 'px';
  }

  /* --------------------------------------------------------------- */
  /* boot                                                             */
  /* --------------------------------------------------------------- */

  function boot() {
    tabPass.addEventListener('click', function () { setMode('passphrase'); });
    tabInvite.addEventListener('click', function () { setMode('invite'); });
    loginForm.addEventListener('submit', doLogin);
    logoutBtn.addEventListener('click', doLogout);
    composer.addEventListener('submit', onSubmit);
    qInput.addEventListener('input', autoGrow);

    // Desktop nicety; on touch keyboards Enter inserts a newline and the
    // send button (thumb-sized) is the primary action.
    qInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { onSubmit(e); }
    });

    window.addEventListener('online',  function () { setBanner(null); });
    window.addEventListener('offline', function () { setBanner('You\u2019re offline. Lu can\u2019t be reached until you reconnect.'); });

    var rec = readToken();
    if (!rec) { dropToken(); showLogin(''); setMode('passphrase'); return; }

    token = rec.t;
    principal = rec.who;
    turns = loadThread();

    // Optimistically show the app, then validate the token cheaply.
    showChat();

    var mine = epoch;
    var handle = {};
    inflight = handle;

    post('/mcp', { method: 'tools/list' },
         { 'Authorization': 'Bearer ' + token }, 20000, handle).then(function (r) {
      if (mine !== epoch) return;
      inflight = null;
      if (r.status === 401) { expire(); setMode('passphrase'); return; }
      if (r.status === 502 || r.status === 503 || r.status === 504) {
        setBanner(COPY.offline); return;
      }
      if (r.networkError) { setBanner(COPY.unreach); return; }
      setBanner(null);
    });
  }

  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('/sw.js').catch(function () { /* non-fatal */ });
    });
  }

  boot();
})();
